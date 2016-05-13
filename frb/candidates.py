from sqlalchemy import (Column, Integer, Float, String, ForeignKey, DateTime)
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import (backref, relation, aliased, sessionmaker)
from sqlalchemy.sql import func


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

    def __init__(self, antenna=None, freq=None, band=None, pol=None,
                 exp_code=None, algo=None, t_0=None, t_end=None, d_t=None,
                 d_nu=None, nu_max=None):
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

    candidate = relation(SearchedData, backref=backref('candidates',
                                                       order_by=id))

    def __init__(self, t, dm):
        """
        :param t:
            Instance of ``astropy.time.Time``.
        :param dm:
            Dispersion measure of pulse.
        """
        self.t = t.utc.datetime
        self.dm = dm

    def __repr__(self):
        return "Candidate. t: {}, DM: {}".format(self.t, self.dm)


# create a connection to a sqlite database
# turn echo on to see the auto-generated SQL
engine = create_engine("sqlite:////home/ilya/code/akutkin/frb/frb/frb.db")

# This creates tables
metadata = Base.metadata
metadata.create_all(engine)

Session = sessionmaker(bind=engine)
session = Session()
session.commit()


def query_frb(exp_code, d_dm, d_t):
    """
    Function that queries DB for FRB events. Any 2 candidates with close DM & t
    values on different antennas will be reported.

    :param exp_code:
        Code of experiment.
    :param d_dm:
        Maximum DM difference allowed for events at different antennas.
    :param d_t:
        Maximum time interval between events at different antennas [s].

    :return:
        List of tuples with ``(t1, t2, dm1, dm2, searched_data1,
        searched_data2)``, where ``searched_data`` - instances of
        ``Searched_data`` classes for corresponding candidates.
    """
    candidates1 = aliased(Candidate)
    candidates2 = aliased(Candidate)
    searched1 = aliased(SearchedData)
    searched2 = aliased(SearchedData)

    query = session.query(candidates1.t, candidates2.t, candidates1.dm,
                          candidates2.dm, searched1, searched2).join((searched1,
                                                candidates1.searched_data_id ==
                                                searched1.id),
                                               (candidates2,
                                                searched1.id ==
                                                candidates2.searched_data_id),
                                               (searched2,
                                                candidates2.searched_data_id ==
                                                searched2.id)).\
        filter(searched2.exp_code == exp_code). \
        filter(searched1.exp_code == exp_code).\
        filter(func.abs(candidates1.dm - candidates2.dm) < d_dm).\
        filter(candidates1.t < candidates2.t). \
        filter(func.abs(func.julianday(candidates2.t) -
                        func.julianday(candidates1.t)) * 86400. < d_t)
    data = query.all()
    for row in data:
        print row

    return data

