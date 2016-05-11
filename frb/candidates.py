from sqlalchemy import Column, Integer, Float, String, ForeignKey
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import backref, mapper, relation, sessionmaker

Base = declarative_base()


class SearchedData(Base):
    """"""
    __tablename__ = "searched_data"

    id = Column(Integer, primary_key=True)
    antenna = Column(String),
    freq = Column(String)
    band = Column(String)
    pol = Column(String)
    exp_code = Column(String)
    algo = Column(String)

    def __init__(self, antenna, freq, band, pol, exp_code, algo):
        self.antenna = antenna
        self.freq = freq
        self.band = band
        self.pol = pol
        self.exp_code = exp_code
        self.algo = algo

    def __repr__(self):
        return "Experiment: {}, antenna: {}, freq: {}," \
               " band: {}, polarization: {}, algo: {}".format(self.exp_code,
                                                              self.antenna,
                                                              self.freq,
                                                              self.band,
                                                              self.pol,
                                                              self.algo)


class Candidate(Base):
    """
    Candidate Class

    Create some class properties before initilization
    """
    __tablename__ = "candidates"
    id = Column(Integer, primary_key=True)
    t = Column(String)
    dm = Column(Float)
    searched_data_id = Column(Integer, ForeignKey('searched_data.id'))

    # creates a bidirectional relationship
    # from Address to SearchedData it's Many-to-One
    # from SearchedData to Address it's One-to-Many
    candidate = relation(SearchedData, backref=backref('candidates',
                                                       order_by=id))

    #----------------------------------------------------------------------
    def __init__(self, t, dm):
        """Constructor"""
        self.t = t
        self.dm = dm

    def __repr__(self):
        return "Candidate. t: {}, DM: {}".format(self.t, self.dm)


# create a connection to a sqlite database
# turn echo on to see the auto-generated SQL
engine = create_engine("sqlite:////home/ilya/code/akutkin/frb/frb/frb.db")
                       # echo=True)

# # get a handle on the table object
# searched_data_table = SearchedData.__table__
# get a handle on the metadata
# This creates tables
metadata = Base.metadata
metadata.create_all(engine)

from sqlalchemy.orm import sessionmaker

Session = sessionmaker(bind=engine)
session = Session()

some_searched_data = SearchedData("AR", "C", "U", "L", "raks3ar", "some_algo")
session.add(some_searched_data)
some_searched_data_too = SearchedData("WB", "L", "U", "L", "raks3az", "some_algo")
session.add(some_searched_data_too)
session.commit()

some_searched_data.candidates = [Candidate("0.01", 500.),
                                 Candidate("0.41", 30.)]
some_searched_data_too.candidates = [Candidate("0.4", 100.)]
session.commit()

all_searched_data = session.query(SearchedData).all()
print all_searched_data
all_candidates = session.query(Candidate).all()
print all_candidates

from sqlalchemy.orm import join
sql = session.query(SearchedData).select_from(join(SearchedData, Candidate))
data = sql.filter(SearchedData.freq == 'L').filter(Candidate.dm > 50.).all()
print data
sql = session.query(Candidate).select_from(join(SearchedData, Candidate))
data = sql.filter(SearchedData.freq == 'L').filter(Candidate.dm > 50.).all()
print data

# sql = session.query(SearchedData, Candidate)
# sql = sql.filter(SearchedData.id == Candidate.searched_data_id)
# sql = sql.filter(SearchedData.antenna == 'AR')
# for u, a in sql.all():
#     print u, a
