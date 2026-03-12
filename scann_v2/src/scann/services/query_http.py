from __future__ import annotations

from typing import Any


def _retry_without_env_proxy(method_name: str, url: str, timeout: int, **kwargs: Any):
    import requests

    session = requests.Session()
    session.trust_env = False
    request_method = getattr(session, method_name)
    try:
        return request_method(url, timeout=timeout, **kwargs)
    finally:
        session.close()


def _request_with_proxy_fallback(method_name: str, url: str, timeout: int, **kwargs: Any):
    import requests

    initial_request = getattr(requests, method_name)
    try:
        return initial_request(url, timeout=timeout, **kwargs)
    except requests.exceptions.ProxyError as exc:
        try:
            return _retry_without_env_proxy(method_name, url, timeout, **kwargs)
        except requests.RequestException as retry_exc:
            raise requests.RequestException(
                f"{exc}; 直连重试失败: {retry_exc}"
            ) from retry_exc


def get_with_proxy_fallback(url: str, timeout: int, **kwargs: Any):
    return _request_with_proxy_fallback("get", url, timeout, **kwargs)


def post_with_proxy_fallback(url: str, timeout: int, **kwargs: Any):
    return _request_with_proxy_fallback("post", url, timeout, **kwargs)


__all__ = ["get_with_proxy_fallback", "post_with_proxy_fallback"]