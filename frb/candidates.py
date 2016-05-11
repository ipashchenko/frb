from sqlalchemy import Column, Integer, Float, String, ForeignKey, DateTime, Interval
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import backref, mapper, relation, sessionmaker

Base = declarative_base()


class SearchedData(Base):
    """
    Class that describes dynamical spectra and it's metadata.
    """
    __tablename__ = "searched_data"

    id = Column(Integer, primary_key=True)
    antenna = Column(String),
    freq = Column(String)
    band = Column(String)
    pol = Column(String)
    exp_code = Column(String)
    algo = Column(String)
    t_0 = Column(DateTime)
    t_end = Column(DateTime)
    d_t = Column(Float)
    d_nu = Column(Float)
    nu_max = Column(Float)

    def __init__(self, antenna, freq, band, pol, exp_code, algo, t_0=None,
                 t_end=None, d_t=None, d_nu=None, nu_max=None):
        self.antenna = antenna
        self.freq = freq
        self.band = band
        self.pol = pol
        self.exp_code = exp_code
        self.algo = algo
        self.t_0 = t_0
        self.t_end = t_end
        self.d_t = d_t
        self.d_nu = d_nu
        self.nu_max = nu_max

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
    Class that describes FRB candidates related to dynamical spectra searched.
    """
    __tablename__ = "candidates"

    id = Column(Integer, primary_key=True)
    t = Column(DateTime)
    dm = Column(Float)
    searched_data_id = Column(Integer, ForeignKey('searched_data.id'))

    # creates a bidirectional relationship
    # from Address to SearchedData it's Many-to-One
    # from SearchedData to Address it's One-to-Many
    candidate = relation(SearchedData, backref=backref('candidates',
                                                       order_by=id))

    def __init__(self, t, dm):
        """Constructor"""
        self.t = t.utc.datetime
        self.dm = dm

    def __repr__(self):
        return "Candidate. t: {}, DM: {}".format(self.t, self.dm)


# create a connection to a sqlite database
# turn echo on to see the auto-generated SQL
engine = create_engine("sqlite:////home/ilya/code/akutkin/frb/frb/frb.db")
                       # echo=True)

# This creates tables
metadata = Base.metadata
metadata.create_all(engine)

from sqlalchemy.orm import sessionmaker

Session = sessionmaker(bind=engine)
session = Session()
session.commit()

# Some examples commented out
# some_searched_data = SearchedData("AR", "C", "U", "L", "raks3ar", "some_algo")
# session.add(some_searched_data)
# some_searched_data_too = SearchedData("WB", "L", "U", "L", "raks3az", "some_algo")
# session.add(some_searched_data_too)
# session.commit()
#
# some_searched_data.candidates = [Candidate("0.01", 500.),
#                                  Candidate("0.41", 30.)]
# some_searched_data_too.candidates = [Candidate("0.4", 100.)]
# session.commit()
#
# all_searched_data = session.query(SearchedData).all()
# print all_searched_data
# all_candidates = session.query(Candidate).all()
# print all_candidates
#
# from sqlalchemy.orm import join
# sql = session.query(SearchedData).select_from(join(SearchedData, Candidate))
# data = sql.filter(SearchedData.freq == 'L').filter(Candidate.dm > 50.).all()
# print data
# sql = session.query(Candidate).select_from(join(SearchedData, Candidate))
# data = sql.filter(SearchedData.freq == 'L').filter(Candidate.dm > 50.).all()
# print data
#
# # sql = session.query(SearchedData, Candidate)
# # sql = sql.filter(SearchedData.id == Candidate.searched_data_id)
# # sql = sql.filter(SearchedData.antenna == 'AR')
# # for u, a in sql.all():
#     print u, a
