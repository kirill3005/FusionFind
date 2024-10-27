import os
import yaml
from sqlalchemy import create_engine, MetaData, Table, select
from sqlalchemy.orm import sessionmaker
from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer
from PIL import Image
import base64
from fastembed import SparseTextEmbedding


class DataMigration:
    def __init__(self, config: dict):
        self.config = config
        self.db_engine, self.db_session = self.connect_to_db()
        self.qdrant_client = self.connect_to_qdrant()
        self.model = SentenceTransformer('intfloat/multilingual-e5-large-instruct')
        self.image_save_path = self.config['image_save_path']  # Путь для сохранения изображений
        self.model_bm42 = SparseTextEmbedding(model_name="Qdrant/bm42-all-minilm-l6-v2-attentions")
        # Создадим директорию для изображений, если она не существует
        if not os.path.exists(self.image_save_path):
            os.makedirs(self.image_save_path)

    # Чтение конфигурационного файла
    def load_config(self, config_path):
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        return config

    # Формирование строки подключения SQLAlchemy
    def create_db_connection_string(self):
        db_config = self.config['database']
        dialect = db_config['dialect']
        host = db_config['host']
        port = db_config['port']
        user = db_config['user']
        password = db_config['password']
        database = db_config['database']

        return f"{dialect}://{user}:{password}@{host}:{port}/{database}"

    # Подключение к базе данных
    def connect_to_db(self):
        connection_string = self.create_db_connection_string()
        engine = create_engine(connection_string)
        Session = sessionmaker(bind=engine)
        session = Session()
        return engine, session

    # Подключение к Qdrant
    def connect_to_qdrant(self):
        qdrant_config = self.config['qdrant']
        client = QdrantClient(host=qdrant_config['host'], port=qdrant_config['port'])
        return client

    # Извлечение данных из базы
    def fetch_data(self, table, columns):
        metadata = MetaData()
        table_obj = Table(table, metadata, autoload_with=self.db_session.bind)

        # Здесь используем распаковку списка с помощью *columns
        query = select(*[table_obj.c[col] for col in columns])

        result = self.db_session.execute(query).fetchall()
        return result

    # Преобразование текста в вектор
    def generate_vector(self, text):
        return self.model.encode(text).tolist()

    def generate_sparse_vector(self, text):
        return list(self.model_bm42.query_embed(text))[0]
    # Сохранение изображения на диск
    def save_image(self, image_data, image_name):
        image_path = os.path.join(self.image_save_path, image_name)
        with open(image_path, 'wb') as img_file:
            img_file.write(base64.b64decode(image_data))
        return image_path

    # Миграция данных в Qdrant
    def migrate(self):
        # Убедимся, что коллекция существует, если нет — создадим
        collection_name = self.config['qdrant']['collection_name']
        vector_size = self.config['qdrant']['vector_size']

        # Определяем конфигурацию для векторов
        vectors_config = {
            "text_vectors": {  # Имя вектора "text_vectors"
                "size": vector_size,
                "distance": "Cosine"
            }
        }
        sparse_vectors_config = {
            "bm42": models.SparseVectorParams(
                modifier=models.Modifier.IDF,
            )
        }
        # Создаем коллекцию (без дополнительных параметров)
        self.qdrant_client.recreate_collection(
            collection_name=collection_name,
            vectors_config=vectors_config,
            sparse_vectors_config=sparse_vectors_config
        )

        # Извлекаем данные из реляционной базы
        table = self.config['mapping']['table']
        vector_column = self.config['mapping']['vector_column']
        metadata_columns = self.config['mapping']['metadata_columns']
        image_column = self.config['mapping'].get('image_column')  # Опционально, столбец с изображением

        data = self.fetch_data(table, [vector_column] + metadata_columns + ([image_column] if image_column else []))

        points = []
        for row in data:
            text = row[0]  # Текст для преобразования в вектор и для текстового поиска
            metadata = {col: val for col, val in zip(metadata_columns, row[1:len(metadata_columns) + 1])}

            # Генерация вектора на основе текста
            vector = self.generate_vector(text)

            sparse_vector = self.generate_sparse_vector(text)

            payload = {
                **metadata,
                'text': text  # Добавляем оригинальный текст в payload для BM42 поиска
            }

            # Если есть изображение, сохраняем его и добавляем путь в payload
            if image_column and row[len(metadata_columns) + 1]:
                image_data = row[len(metadata_columns) + 1]
                image_name = f"{metadata['id']}.png"
                image_path = self.save_image(image_data, image_name)
                payload['image_path'] = image_path

            # Формируем точку (point) для вставки в коллекцию
            points.append({
                'id': metadata['id'],  # Уникальный идентификатор (например, id)
                'vector': {'text_vectors': vector, 'bm42': models.SparseVector(values = sparse_vector.values.tolist(), indices=sparse_vector.indices.tolist())},  # Векторная часть для поиска по вектору
                'payload': payload,  # Метаданные и текст для поиска по BM42
            })

        # Загружаем данные в Qdrant через метод upsert
        self.qdrant_client.upsert(
            collection_name=collection_name,
            points=points
        )

        # Добавление текстового индекса для поиска
        self.qdrant_client.create_payload_index(
            collection_name=collection_name,
            field_name="text",  # Поле для индексирования
            field_type="text"  # Указываем, что это текстовое поле
        )

        print(f"Миграция завершена. Загружено {len(points)} записей.")
        self.db_session.close()

    def search_sparse(self, query, limit):

        sparse_vector_fe = list(self.model_bm42.query_embed(query))[0]

        sparse_vector = models.SparseVector(
            values=sparse_vector_fe.values.tolist(),
            indices=sparse_vector_fe.indices.tolist()
        )

        result = self.qdrant_client.query_points(
            collection_name=self.config['qdrant']['collection_name'],
            query=sparse_vector,
            using="bm42",
            with_payload=True,
            limit=limit
        )

        return result.points

    def find_nearest_embedding(self, query_text):
        """
        Выполняет комбинированный поиск: сначала по тексту (BM42), затем по вектору запроса для уточнения результатов.

        :param query_text: Текст запроса для генерации вектора
        :return: Самый близкий результат по вектору среди топ-25 по текстовому поиску с полезной нагрузкой
        """
        # Шаг 1: Поиск с помощью BM42 для выбора топ-25 результатов
        bm42_results = self.search_sparse(query_text, 25)

        # Проверяем, нашлись ли результаты
        if not bm42_results:
            print("Результаты BM42 не найдены.")
            return None

        # Извлекаем IDs результатов BM42 для дальнейшего фильтра векторного поиска
        bm42_ids = [result.id for result in bm42_results]

        # Шаг 2: Генерируем вектор запроса
        query_vector = self.generate_vector(query_text)

        # Шаг 3: Выполняем векторный поиск среди результатов, найденных на этапе BM42
        vector_search_result = self.qdrant_client.search(
            collection_name=self.config['qdrant']['collection_name'],
            query_vector={"name": "text_vectors", "vector": query_vector},  # Указываем имя вектора и сам вектор
            limit=1,  # Ищем один самый близкий вектор
            query_filter={
                "must": [
                    {
                        'has_id': bm42_ids
                    }
                ]
            },
            with_payload=True  # Возвращаем полезную нагрузку (payload)
        )

        # Проверяем, есть ли результаты векторного поиска
        if not vector_search_result:
            print("Результаты векторного поиска не найдены.")
            return None

        # Извлекаем ближайшую точку
        nearest_point = vector_search_result[0]

        # Возвращаем результат
        return nearest_point








