from sqlalchemy.ext.declarative import declarative_base, as_declarative
from sqlalchemy import ForeignKey, UniqueConstraint, Column, Integer, String, Text, Date, DateTime, Float, Numeric, DECIMAL, inspect, INT
from sqlalchemy_utils import EmailType, PasswordType, PhoneNumberType, ChoiceType, CurrencyType, Currency

Base = declarative_base()

class SummarizeLog(Base):
    __tablename__ = "summarize_log"

    prediction_id = Column(String(length = 10), primary_key = True, nullable = False) # primary key
    prediction_ts = Column(DateTime(), nullable = False)  # date of creation
    document = Column(String(length = 10000), nullable = False)
    summary = Column(String(length = 1000), nullable = False)
    reference_summary = Column(String(length = 1000), nullable = False)
    model_source = Column(String(length=20), nullable = False) # source be {local, mlflow, lambda, sagemaker}
    score_blue = Column(Float(), nullable = True)
    score_rouge1 = Column(Float(), nullable = True)
    score_rouge2 = Column(Float(), nullable = True)
    score_rougeL = Column(Float(), nullable = True)

class SummarizeSample(Base):
    __tablename__ = "summarize_sample"

    sample_id = Column(String(length = 10), primary_key = True, nullable = False)
    article = Column(String(length = 10000), nullable = False)
    summ = Column(String(length = 1000), nullable = False)