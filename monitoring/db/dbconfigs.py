import logging
from sqlalchemy import create_engine, and_
from sqlalchemy.orm import sessionmaker, Session, Query
from sqlalchemy.inspection import inspect


#######################################
###  Base Database Configuration    ###
#######################################

class baseDbInf:
    def __init__(self, driver: str):
        """the base configuration object, can be inherent by MySQL, DB2, SQLite, etc.

        :param driver: the driver to use, string value
        """
        self.driver = driver

    def bindServer(self, ip: str, port: int = None, db: str = None):
        """Connect to a database server

        :param ip: ip of the server
        :param port: port number of the server
        :param db: which database to connect
        :return:
        """
        self.ip = ip
        self.port = port
        self.db = db

    def login(self, username: str, password: str):
        """Login to server-client based database (MySQL, DB2, SqlServer, Hive, etc.)

        :param username:
        :param password:
        :return:
        """
        self.username = username
        self.password = password

    def argTempStr(self):
        raise NotImplementedError("Must implement this method to get different arg placeholder for different database")

    def getConnStr(self) -> str:
        """Connection string for sqlalchemy

        :return:
        """
        #engine = create_engine('mysql+mysqlconnector://USRNAME:PSWD@localhost:3306/DATABASE?charset=ytf8')
        #engine = create_engine("ibm_db_sa://USRNAME:PSWD@IP:PORT/DATABASE?charset=utf8")
        #engine = create_engine('sqlite:///DB_ADDRESS')
        return f"{self.driver}://{self.username}:{self.password}@{self.ip}:{self.port}/{self.db}?charset=utf8"

    def launch(self):
        """Launch the databse connector, create the sqlalchemy engine and create a session

        :return:
        """
        connStr = self.getConnStr()
        self.engine = create_engine(connStr)
        self.DBSession = sessionmaker(bind = self.engine)
        logging.info("Engine started, ready to go!")

    def newSession(self):
        try:
            session = self.DBSession()
        except Exception as e:
            logging.error(e)
        else:
            return session

    def getJdbcUrl(self) -> str:
        """Get JDBC connection string, for spark connection and other purpose

        :return:
        """
        raise NotImplementedError("Must implement this method to get different JdbcUrl for different database")

    def getDriverClass(self) -> str:
        # for spark connection and other purpose
        raise NotImplementedError("Must implement this method to get different DriverClass for different database")


#############################################
### Provider level database Configuration ###
#############################################


class MySQL(baseDbInf):
    def __init__(self, driver = "mysqlconnector"):
        super().__init__(f"mysql+{driver}")

    def argTempStr(self):
        return "%s"

    def getJdbcUrl(self) -> str:
        return f"jdbc:mysql://{self.ip}:{self.port}/{self.db}"

    def getDriverClass(self) -> str:
        # https://repo1.maven.org/maven2/mysql/mysql-connector-java/8.0.27/mysql-connector-java-8.0.27.jar
        return "com.mysql.jdbc.Driver"