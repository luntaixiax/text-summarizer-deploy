import logging
import pandas as pd
import numpy as np
import contextlib
from sqlalchemy import create_engine, and_
from sqlalchemy.orm import sessionmaker, Session, Query
from sqlalchemy.inspection import inspect
from db.dbconfigs import baseDbInf


#######################################
### Database accesor Object -- DAO  ###
#######################################

class dbIO:

    def __init__(self, dbinf: baseDbInf):
        """database accessor designed for pandas and sqlalchemy

        :param dbinf: database configuration object (DB2, MySQL, etc.)
        """
        dbinf.launch()
        self.__dbinf = dbinf
        self.placeholder = dbinf.argTempStr()

    def getDbInf(self):
        return self.__dbinf

    def getEngine(self):
        return self.getDbInf().engine

    @ contextlib.contextmanager
    def get_session(self, errormsg : str = "reason Unknown") -> Session:
        """return the session object to operate

        :param errormsg: the error message to be displayed if error occured
        :return: Session object

        example:
        >>> with self.get_session() as s:
        >>>     #do something here
        """
        session = self.getDbInf().newSession()
        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            logging.error("Error Occured: %s\n%s" % (e.args, errormsg))
        finally:
            session.close()

    def insert(self, table, record : dict) -> None:
        """insert a record into db

        :param table: table class
        :param record: dict of record to insert
        :return: None

        example:
        >>> x = {"fund_id" : "160023", "date" : "2005-09-02", "net_value" : 1.0102, "full_value" : 3.0102, "div" : 0.234, "pnl" : 0.2341}
        >>> dbIO.insert(MutualFundHist, x)
        """
        with self.get_session() as s:
            s.add(table(**record))

    def update(self, table, primary_kvs : dict, record : dict) -> None:
        """update record by looking at primary key

        :param table: table class
        :param primary_kvs: dict of primary key-value pairs, use to find which record(s) to update
        :param record: dict of new record
        :return: None

        example:
        >>> p = {"fund_id" : "150234", "date" : "2005-09-09"}  # use to find which records to update_all
        >>> r = {"net_value" : 0.4456}  # the new record to save to db
        >>> dbIO.update(MutualFundHist, p, r)
        is equivalent to:
        UPDATE MutualFundHist
        SET net_value = 0.4456
        WHERE fund_id = '150234' AND date = '2005-09-09';
        """
        with self.get_session() as s:
            #s.query_extract(table).filter(table.fund_id == "150234").update_all(r)
            conditions = [getattr(table, k) == v for k, v in primary_kvs.items()]
            s.query(table).filter(and_(*conditions)).update_all(record)

    def delete(self, table, primary_kvs : dict) -> None:
        """delete record by looking at primary key

        :param table: table class
        :param primary_kvs: dict of primary key-value pairs, use to find which record(s) to update_all
        :return: None

        example:
        >>> p = {"fund_id" : "150234", "date" : "2005-09-09"}  # use to find which records to update_all
        >>> self.delete(MutualFundHist, p)
        is equivalent to:
        DELETE FROM MutualFundHist
        WHERE fund_id = '150234' AND date = '2005-09-09';
        """
        with self.get_session() as s:
            conditions = [getattr(table, k) == v for k, v in primary_kvs.items()]
            s.query(table).filter(and_(*conditions)).delete()


    def modify_sql(self, sql : str, *args, errormsg : str = "reason Unknown", **kws) -> int:
        """execute original sql (not query_extract)

        :param sql: sql string
        :param args: positional arguments to fill in sql template
        :param kws: kw arguments to fill in sql template
        :param errormsg: error message to display when error occurred
        :return: 1: success, 0: failed

        example:
        >>> sql = 'UPDATE MutualFundHist SET net_value = %s WHERE fund_id = %s AND date = %s;'
        >>> self.modify_sql(sql, 0.7456, '150234', '2005-09-09')
        """
        with self.getEngine().connect() as conn:
            try:
                sql = sql.replace("?", self.placeholder)
                conn.execute(sql, *args, **kws)
            except Exception as e:
                logging.error("Error Occured: %s\n%s" % (e.args, errormsg))
                return 0
            else:
                return 1

    def query_df(self, query : Query) -> pd.DataFrame:
        """making query_extract directly

        :param query: Query statement
        :return: resulting dataframe

        example:
        >>> with self.get_session() as s:
        >>>     query_extract = s.query_extract(MutualFundHist).filter(MutualFundHist.fund_id == "160023")
        >>>
        >>> df = self.query_df(query_extract)
        """
        return pd.read_sql(query.statement, query.session.bind).replace({None : np.nan})

    def query_sql_df(self, sql : str, *args, errormsg : str = "reason Unknown", **kws) -> pd.DataFrame:
        """using original sql to make queries

        :param sql: the sql template
        :param args: the argument list to fill in the sql template
        :param errormsg: message to display when encoutering errors
        :param kws: the argument kws to fill in the sql template
        :return: the result dataframe

        example:
        >>> sql = "select * from mutualfundhist where fund_id = ?"
        >>> r = self.query_sql_df(sql, '160023')
        """
        with self.getEngine().connect() as conn:
            try:
                sql = sql.replace("?", self.placeholder)
                r = conn.execute(sql, *args, **kws)
                #df = pd.read_sql(sql, conn, params = args)
            except Exception as e:
                logging.error("Error Occured: %s\n%s" % (e.args, errormsg))
                df = pd.DataFrame({})
            else:
                if hasattr(r, "cursor"):
                    description = r.cursor.description
                else:
                    description = r.description

                headers = [ i[0] for i in description ]
                # df = pd.DataFrame.from_records(r, columns = headers)
                df = pd.DataFrame.from_records(r, columns = headers, coerce_float = True)
            finally:
                return df

    def insert_df(self, table, df : pd.DataFrame) -> None:
        """save a pandas table into database

        :param table: table class
        :param df: the pandas Dataframe to save into db
        :return:

        example:
        >>> df = pd.DataFrame({
        >>>     "fund_id" : ["160023", "160023", "160023", "150234"],
        >>>     "date" : ["2005-09-09", "2005-09-02", "2005-08-31", "2005-09-09"],
        >>>     "net_value" : [1.0234, 1.0102, 1.0456, 0.9876],
        >>>     "full_value" : [3.0234, 3.0102, 3.0456, 2.9876],
        >>>     "div" : [None, 0.234, None, None],
        >>>     "split_ratio" : [1.0034, None, None, 1.2232],
        >>>     "pnl" : [0.2334, 0.2341, -0.1442, -0.0032],
        >>> })
        >>> self.insert_df(MutualFundHist, df)
        """
        records = df2Tables(df, table)

        with self.get_session() as s:
            s.add_all(records)

    def insert_pd_df(self, tablename: str, df : pd.DataFrame, schema: str = None, mode: str = "append", index: bool = False,
                     chunksize: int = 1000, method: str = 'multi', errormsg : str = "reason Unknown") -> int:
        """save a pandas dataframe to database using pandas style

        :param tablename: the name str of the table
        :param df: the pandas dataframe to insert
        :param schema: the schema you are going to drop table into
        :param mode: {‘fail’, ‘replace’, ‘append’}, default ‘append’
        :param index: whether to include the index
        :param chunksize: Specify the number of rows in each batch to be written at a time. recommend 1000 or None
        :param method: {None, ‘multi’, callable}
        :return: 1: success, 0: failed

        check here: https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.to_sql.html
        speed up insert:
        use method = multi will significantly speed up as it will insert in a batch
        limit chunk size in accordance with the limit of your database
        """
        try:
            with self.getEngine().begin() as conn:  # .begin() will automatically tackle with rollback() and commit()
                df.to_sql(tablename, conn, schema = schema, if_exists = mode, index = index, chunksize = chunksize, method = method)
        except Exception as e:
            logging.error("Error Occured: %s\n%s" % (e.args, errormsg))
            return 0
        else:
            return 1


    def insert_sql_df(self, tablename: str, df : pd.DataFrame, errormsg : str = "reason Unknown") -> int:
        """save a pandas dataframe to database using pythonic style

        :param tablename: the name str of the table
        :param df: the pandas dataframe to insert
        :return: 1: success, 0: failed
        """
        sql = f"""
        insert into {tablename} ({",".join(df.columns)})
        values ({",".join("?" for col in df.columns)})
        """
        sql = sql.replace("?", self.placeholder)

        conn = self.getEngine().raw_connection()
        cursor = conn.cursor()
        try:
            cursor.executemany(sql, df.values)
        except Exception as e:
            conn.rollback()
            logging.error("Error Occured: %s\n%s" % (e.args, errormsg))
            return 0
        else:
            conn.commit()
            return 1
        finally:
            cursor.close()
            conn.close()



def getPrimaryKeys(table) -> list:
    return [key.name for key in inspect(table).primary_key]

def df2Tables(df : pd.DataFrame, table) -> list:
    records = []
    for row in df.replace({np.nan: None}).to_dict(orient = "records"):
        records.append(table(**row))

    return records

#######################################
### JDBC drivers download and setup ###
#######################################


if __name__ == '__main__':


    pass