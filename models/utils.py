def cleanNullValues(json_obj):
    if isinstance(json_obj, dict) and json_obj:
        clean = {}
        for k, v in json_obj.items():
            nested = cleanNullValues(v)
            if nested is not None:
                clean[k] = nested
        return clean
    elif isinstance(json_obj, list) and json_obj:
        return [cleanNullValues(v_) for v_ in json_obj if v_ is not None]
    elif json_obj is not None:
        return json_obj
    
    return None

def truncate_long_strings(data, max_string_length=10_000):
    """Recursively truncate all strings to max_string_length"""
    if isinstance(data, dict):
        for k, v in data.items():
            data[k] = truncate_long_strings(v, max_string_length)
    elif isinstance(data, list):
        for i, v in enumerate(data):
            data[i] = truncate_long_strings(v, max_string_length)
    elif isinstance(data, str) and len(data)>max_string_length:
        return data[:max_string_length]
    return data


def truncate_long_arrays(data, max_array_length=30):
    """Recursively truncate all arrays to max_array_length"""
    if isinstance(data, dict):
        for k, v in data.items():
            data[k] = truncate_long_arrays(v)
    elif isinstance(data, list):
        if len(data) > max_array_length:
            data = data[:max_array_length]
        for i, v in enumerate(data):
            data[i] = truncate_long_arrays(v)
    return data

def truncate_data_dfs(data, max_length=100_000):
    """Truncate data to fit max_length"""
    total_length = 0
    if isinstance(data, dict):
        for k, v in data.items():
            if total_length > max_length:
                data[k] = None
            else:
                data[k], length = truncate_data_dfs(v, max_length=max_length - total_length)
                total_length += length
        return data, total_length

    if isinstance(data, list):
        for i, v in enumerate(data):
            if total_length > max_length:
                data[i] = None
            else:
                data[i], length = truncate_data_dfs(v, max_length=max_length - total_length)
                total_length += length
        return data, total_length

    if isinstance(data, str):
        data_str = data
    else:
        data_str = str(data)
    return data, len(data_str)
