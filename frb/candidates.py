# -*- coding: utf-8 -*-
from sqlalchemy import Column, Integer, String, ForeignKey
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import backref, mapper, relation, sessionmaker


Base = declarative_base()

# TODO: Mapping preprocessing & search function - algorithm code
class Candidate(Base):
    """
    Class tha describes FRB candidate.

    :param t:
        Estimated time of arrival.
    :param dm:
        Estimated DM.
    :param antenna:
        Antenna used.
    :param freq:
        Frequency ['p', 'l', 'c'].
    :param band:
        Band ['u', 'l']
    :param pol:
        Polarization ['l', 'r'].
    :param exp_code:
        Experiment code.
    :param algo:
        String of algorithm specification.
    """
    __tablename__ = "candidates"

    id = Column(Integer, primary_key=True),
    antenna = Column(String),
    time = Column(String),
    dm = Column(String)
    freq = Column(String)
    band = Column(String)
    pol = Column(String)
    exp_code = Column(String)
    algo = Column(String)

    def __init__(self, t, dm, antenna, freq, band, pol, exp_code, algo):
        self.t = t
        self.dm = dm
        self.antenna = antenna
        self.freq = freq
        self.band = band
        self.pol = pol
        self.exp_code = exp_code
        self.algo = algo

    def __cmp__(self, other):
        """
        Compare candidate with another candidate (from other antenna)
        :param other:
            Instance of ``Candidate`` class.
        :return:
            ``True`` if candidates are close in time and DM.
        """
        raise NotImplementedError

# create a connection to a sqlite database
# turn echo on to see the auto-generated SQL
engine = create_engine("sqlite:///tutorial.db", echo=True)

# get a handle on the table object
candidates_table = Candidate.__table__
# get a handle on the metadata
metadata = Base.metadata
metadata.create_all(engine)


from sqlalchemy.orm import sessionmaker

Session = sessionmaker(bind=engine)
session = Session()

candidate = Candidate("134134134.1234", "134.5", "AR", "654514.0", "U", "L",
                      "rk02hr", "dd1search2")
session.add(candidate)
session.add_all([Candidate("134134134.1235", "144.5", "AR", "654514.0", "U", "L",
                      "rk02hr", "dd1search5"),
                 Candidate("134134134.1234", "134.5", "AR", "654514.0", "U", "L",
                      "rk02hr", "dd1search2")])
session.commit()

