import datetime
from enum import Enum

import numpy as np


class TimeTool:
    class DateFormat(Enum):
        ClassDate = 1
        YearDay = 2
        MJD = 3
        YMD = 4
        YearFraction = 5
        TimeDelta = 6
        YMD_dash = 7

    @staticmethod
    def convert_date_format(date, input_type: DateFormat = DateFormat.ClassDate,
                            output_type: DateFormat = DateFormat.YearFraction,
                            from_date: datetime.date = None):
        """
        get the date in format of datetime.date
        input/output_type restrict the type or format of param date and the return.
        DateFormat.ClassDate -> datetime.date.
        DateFormat.YearDay -> str in format of 'yyyyddd', y: year; d: the d-th day of this year. e.g., '2002031'.
        DateFormat.MJD -> int, Modified Julian Day.
        DateFormat.YMD -> str in format of 'yyyymmdd', y: year; m:month; d: day. e.g., '20020131'.
        DateFormat.YMD_dash -> str in format of 'yyyy-mm-dd', y: year; m:month; d: day. e.g., '2002-01-31'.
        DateFormat.TimeDelta -> int, days from from_date.
        :return:
        """

        def _convert_to_class_date(d, i_type: TimeTool.DateFormat):

            if i_type == TimeTool.DateFormat.ClassDate:
                result = d

            elif i_type == TimeTool.DateFormat.YearDay:
                d_str = str(d)
                assert len(d_str) == 7
                year_str = d_str[:4]
                days_str = d_str[4:]

                result = datetime.date(int(year_str), 1, 1) + datetime.timedelta(int(days_str) - 1)

            elif i_type == TimeTool.DateFormat.MJD:
                d0 = datetime.date(1858, 11, 17)
                result = d0 + datetime.timedelta(days=int(d))

            elif i_type == TimeTool.DateFormat.YMD:
                d_str = str(d)
                assert len(d_str) == 8
                year_str = d_str[:4]
                month_str = d_str[4:6]
                day_str = d_str[6:]

                result = datetime.date(int(year_str), int(month_str), int(day_str))

            elif i_type == TimeTool.DateFormat.YMD_dash:
                d_str = str(d)
                assert len(d_str) in (8, 9, 10)
                ymd = d_str.split("-")
                assert len(ymd) == 3

                year_str = ymd[0]
                month_str = ymd[1]
                day_str = ymd[2]

                result = datetime.date(int(year_str), int(month_str), int(day_str))

            elif i_type == TimeTool.DateFormat.YearFraction:
                year = int(d)
                if TimeTool.is_leap(year):
                    days_of_year = int((d - year) * 366)
                else:
                    days_of_year = int((d - year) * 365)

                year_day = year * 1000 + days_of_year

                return _convert_to_class_date(year_day, i_type=TimeTool.DateFormat.YearDay)

            elif i_type == TimeTool.DateFormat.TimeDelta:
                assert from_date is not None
                return from_date + datetime.timedelta(days=int(d))

            else:
                raise Exception

            return result

        def _convert_from_class_date_to(d: datetime.date, o_type: TimeTool.DateFormat):
            year = d.year
            month = d.month
            day = d.day

            if o_type == TimeTool.DateFormat.ClassDate:
                result = d

            elif o_type == TimeTool.DateFormat.YearDay:
                time_delta = d - datetime.date(year, 1, 1)
                dth_day = time_delta.days + 1
                dth_day_str = str(dth_day).rjust(3, '0')

                result = f'{str(year)}{dth_day_str}'

            elif o_type == TimeTool.DateFormat.MJD:
                timedelta = d - datetime.date(1858, 11, 17)
                result = timedelta.days

            elif o_type == TimeTool.DateFormat.YMD:
                year_str = str(year)
                month_str = str(month).rjust(2, '0')
                day_str = str(day).rjust(2, '0')
                result = f'{year_str}{month_str}{day_str}'

            elif o_type == TimeTool.DateFormat.YMD_dash:
                year_str = str(year)
                month_str = str(month).rjust(2, '0')
                day_str = str(day).rjust(2, '0')
                result = f'{year_str}-{month_str}-{day_str}'

            elif o_type == TimeTool.DateFormat.YearFraction:
                days_of_year = (d - datetime.date(year, 1, 1)).days + 0.5

                if TimeTool.is_leap(year):
                    return year + days_of_year / 366

                else:
                    return year + days_of_year / 365

            elif o_type == TimeTool.DateFormat.TimeDelta:
                assert from_date is not None
                return (d - from_date).days

            else:
                raise Exception

            return result

        # if type(date) is list:
        if len(np.shape(date)) == 1:
            d_class_date = [_convert_to_class_date(date[i], i_type=input_type) for i in range(len(date))]
            d_target_type = [_convert_from_class_date_to(d_class_date[i], o_type=output_type) for i in
                             range(len(d_class_date))]

        else:
            d_class_date = _convert_to_class_date(date, i_type=input_type)
            d_target_type = _convert_from_class_date_to(d_class_date, o_type=output_type)

        return d_target_type

    @staticmethod
    def get_the_final_day_of_this_month(date: datetime.date = None, year: int = None, month: int = None):
        input_date = date is not None

        input_year = year is not None
        input_month = month is not None

        assert input_year == input_month
        assert input_date ^ input_year

        if input_date:
            year = date.year
            month = date.month
        else:
            pass

        if month == 12:
            return datetime.date(year, month, 31)

        else:
            return datetime.date(year, month + 1, 1) - datetime.timedelta(1)

    @staticmethod
    def is_leap(year):
        if (year % 4 == 0) and (year % 100 != 0 or year % 400 == 0):
            return True
        else:
            return False

    @staticmethod
    def get_average_dates(begin, end, unused=None):
        """
        Obtain the average date by entering the start and end date(s).
        The start/end date(s) can be a datetime.date object, or a list composed of it with consistent length,
        and the return value can also be a datetime.date object or a list composed of it.

        The optional parameter 'unused' is supposed to be a list, or a list nested within a list,
        whose elements should also be a datetime.date object.
        If there are parameters passed in of parameter 'unused',
        these dates will not be considered when calculating the average.
        """

        assert type(begin) in (datetime.date, list)
        assert type(end) in (datetime.date, list)
        assert type(end) is type(begin)

        if type(begin) is datetime.date:
            if unused is None:
                time_delta = end - begin

                ave_date = begin + time_delta / 2
                return ave_date

            else:
                '''get used dates'''
                used_days = []
                max_days = (end - begin).days
                for i in range(max_days + 1):
                    this_date = begin + datetime.timedelta(days=i)
                    if this_date not in unused:
                        used_days.append(this_date)

                '''get averaged dates of used dates'''
                mjd = 0
                for i in range(len(used_days)):
                    mjd += TimeTool.convert_date_format(
                        used_days[i],
                        input_type=TimeTool.DateFormat.ClassDate,
                        output_type=TimeTool.DateFormat.MJD
                    )

                return TimeTool.convert_date_format(
                    mjd / len(used_days),
                    input_type=TimeTool.DateFormat.MJD,
                    output_type=TimeTool.DateFormat.ClassDate
                )

        else:
            ave_dates = []
            for i in range(len(begin)):
                ave_date = TimeTool.get_average_dates(begin[i], end[i], unused[i] if type(unused) is list else None)

                ave_dates.append(ave_date)

            return ave_dates
