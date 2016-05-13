from sqlalchemy import create_engine
from sqlalchemy.orm import (aliased, sessionmaker)
from sqlalchemy.sql import func
from candidates import (Candidate, SearchedData)


def connect_to_db(db_file):
    engine = create_engine("sqlite:///{}".format(db_file))
    Session = sessionmaker(bind=engine)
    session = Session()
    return session


def query_frb(session, exp_code, d_dm, d_t):

    sq = session.query(Candidate, SearchedData).select_from(Candidate).\
        join(SearchedData, SearchedData.id == Candidate.searched_data_id). \
        filter(SearchedData.exp_code == exp_code).subquery()
    sq2 = aliased(sq)
    result = session.query(sq.c.t, sq2.c.t, sq.c.dm, sq2.c.dm, sq.c.antenna,
                           sq2.c.antenna).\
        join(sq2, sq.c.exp_code == sq2.c.exp_code). \
        filter(sq.c.antenna != sq2.c.antenna). \
        filter(func.abs(sq.c.dm - sq2.c.dm) < d_dm). \
        filter(sq.c.t < sq2.c.t). \
        filter(func.abs(func.julianday(sq2.c.t) -
                        func.julianday(sq.c.t)) * 86400. < d_t).all()
    return result


if __name__ == '__main__':
    session = connect_to_db('/home/ilya/code/akutkin/frb/frb/frb.db')
    frb_list = query_frb(session, 'raks00', 100, 0.1)
    for frb in frb_list:
        print frb

