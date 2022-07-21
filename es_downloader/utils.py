from typing import Any, Dict, List, Optional, Union


def remove_empty_dict_values(dic: Dict[str, Any]) -> Dict[str, Any]:
    """Remove empty values inside a dict."""
    return {k: v for k, v in dic.items() if v is not None}


def clean_dict_values(dic: Dict[str, Any]) -> Dict[str, Any]:
    """Convert booleans and lists to strings in a dict."""
    for key, value in dic.items():

        if isinstance(value, bool):
            # convert a boolean to a string
            dic[key] = str(value).lower()

        elif isinstance(value, list):
            # convert a list to a string
            dic[key] = ",".join([str(i) for i in value])

    return dic


def clean_params(params: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Clean requests params removing empty values."""
    if not params:
        return None
    params = remove_empty_dict_values(params)
    params = clean_dict_values(params)
    return params
