import re


def n_distinct(list1):
    unique_list = []
    for x in list1:
        if x not in unique_list:
            unique_list.append(x)
    return len(unique_list)


def check_second(timestamp_list):
    return 1 if n_distinct(timestamp_list.second) > 1 else 0


def check_minute(timestamp_list):
    return 1 if n_distinct(timestamp_list.minute) > 1 else 0


def check_hour(timestamp_list):
    return 0
    #return 1 if n_distinct(timestamp_list.hour) > 1 else 0


def check_days(timestamp_list):
    return 1 if abs((timestamp_list[0] - timestamp_list[-1]).days) > 30 else 0


def check_week(timestamp_list):
    return 1 if n_distinct(timestamp_list.month) > 4 else 0


def check_month(timestamp_list):
    return 1 if n_distinct(timestamp_list.month) > 11 else 0


def check_year(timestamp_list):
    return 1 if n_distinct(timestamp_list.year) > 5 else 0


def interpret_timestamp(timestamp_list):
    """
    :param timestamp_list: dataframe index with date type values
    :return: bitmap of granularities that should be explored
        [minute, hour, day, week, month, year]
    """
    date = timestamp_list[0]
    yy_mm_dd_HH_MM_SS_pattern = "^[1-2]{1}[0-9]{3}[-]{1}[0-9]{2}[-]{1}[0-9]{2} [0-9]{2}:[0-9]{2}:[" \
                                "0-9]{2}[.]*[0-9]*[a-z]*$"
    yy_mm_dd_HH_MM_pattern = "^[1-2]{1}[0-9]{3}[-]{1}[0-9]{2}[-]{1}[0-9]{2} [0-9]{2}:[0-9]{2}$"
    year_month_day_hour_pattern = "^[1-2]{1}[0-9]{3}[-]{1}[0-9]{2}[-]{1}[0-9]{2} [0-9]{2}$"
    year_month_day_pattern = "^[1-2]{1}[0-9]{3}[-]{1}[0-9]{2}[-]{1}[0-9]{2}$"
    year_month_pattern = "^[1-2]{1}[0-9]{3}[-]{1}[0-9]{2}$"
    year_pattern = "^[1-2]{1}[0-9]{3}$"

    if re.findall(yy_mm_dd_HH_MM_SS_pattern, str(date)):
        return [check_second(timestamp_list),
                check_minute(timestamp_list),
                check_hour(timestamp_list),
                check_days(timestamp_list),
                check_week(timestamp_list),
                check_month(timestamp_list),
                check_year(timestamp_list)]
    elif re.findall(yy_mm_dd_HH_MM_pattern, str(date)):
        return [0,
                check_minute(timestamp_list),
                check_hour(timestamp_list),
                check_days(timestamp_list),
                check_week(timestamp_list),
                check_month(timestamp_list),
                check_year(timestamp_list)]
    elif re.findall(year_month_day_hour_pattern, str(date)):
        return [0,
                0,
                check_hour(timestamp_list),
                check_days(timestamp_list),
                check_week(timestamp_list),
                check_month(timestamp_list),
                check_year(timestamp_list)]
    elif re.findall(year_month_day_pattern, str(date)):
        return [0,
                0,
                0,
                0,
                check_week(timestamp_list),
                check_month(timestamp_list),
                check_year(timestamp_list)]
    elif re.findall(year_month_pattern, str(date)):
        return [0,
                0,
                0,
                0,
                0,
                check_month(timestamp_list),
                check_year(timestamp_list)]
    elif re.findall(year_pattern, str(date)):
        return [0,
                0,
                0,
                0,
                0,
                0,
                check_year(timestamp_list)]
    else:
        raise ValueError('Timestamp format incorrect')
