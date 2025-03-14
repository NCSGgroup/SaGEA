import datetime

from pysrc.auxiliary.aux_tool.TimeTool import TimeTool


class DateSeries:
    def __init__(self, dates: list[datetime.date] or datetime.date):
        if type(dates) is datetime.date:
            dates = [dates]

        assert type(dates) is list

        self.__value = dates

    def get_dates(self, in_format: TimeTool.DateFormat = None) -> list[datetime.date]:
        if in_format is None:
            in_format = TimeTool.DateFormat.ClassDate

        assert type(in_format) is TimeTool.DateFormat

        result = TimeTool.convert_date_format(
            self.__value, input_type=TimeTool.DateFormat.ClassDate, output_type=in_format
        )

        return result

    def get_length(self) -> int:
        return len(self)

    def __len__(self) -> int:
        return len(self.__value)

    def append(self, date: datetime.date):
        self.__value.append(date)

        return None


if __name__ == '__main__':
    ds = DateSeries([datetime.date(2020, 1, 1), datetime.date(2020, 1, 2)])

    print(ds.get_length())
