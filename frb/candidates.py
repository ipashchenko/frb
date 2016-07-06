# -*- coding: utf-8 -*-
import os
from sqlalchemy import (Column, Integer, Float, String, ForeignKey, DateTime)
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import (backref, relation, sessionmaker)


Base = declarative_base()


class SearchedData(Base):
    """
    Class that describes dynamical spectra and it's metadata.
    """
    __tablename__ = "searched_data"

    id = Column(Integer, primary_key=True)
    antenna = Column(String)
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
        return "Experiment: {}, antenna: {}, time begin: {}, time end: {}," \
               "freq: {}, band: {}, polarization: {}," \
               " algo: {}".format(self.exp_code, self.antenna, self.t_0,
                                  self.t_end, self.freq, self.band, self.pol,
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
        # return "FRB candidate. t0: {}, DM: {}".format(self.t, self.dm)
        return "FRB candidate. t0: " \
               "{:%Y-%m-%d %H:%M:%S.%f}".format(self.t)[:-3] +\
               " DM: {:.0f}".format(self.dm)


# create a connection to a sqlite database
db_file = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                       'frb.db')
engine = create_engine("sqlite:///{}".format(db_file))

# This creates tables
metadata = Base.metadata
metadata.create_all(engine)

Session = sessionmaker(bind=engine)
session = Session()
session.commit()
