# -*- coding: utf-8 -*-


class Candidate(object):
    def __init__(self, t, dm, antenna, meta_data):
        self.t = t
        self.dm = dm
        self.antenna = antenna
        self.meta_data = meta_data

    def save_to_db(self):
        """
        Save candidate to candidates DB.
        """
        raise NotImplementedError

    def __cmp__(self, other):
        """
        Compare candidate with another candidate (from other antenna)
        :param other:
            Instance of ``Candidate`` class.
        :return:
            ``True`` if candidates are close in time and DM.
        """
        raise NotImplementedError