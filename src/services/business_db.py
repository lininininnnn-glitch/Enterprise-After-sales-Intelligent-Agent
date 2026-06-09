"""本地业务数据模拟。"""

from __future__ import annotations

from typing import Dict

_USERS: Dict[str, dict] = {
    "U1001": {"user_id": "U1001", "name": "张三", "vip_level": "黄金VIP", "city": "上海"},
    "U2002": {"user_id": "U2002", "name": "李四", "vip_level": "普通用户", "city": "深圳"},
}

_ORDERS: Dict[str, dict] = {
    "888": {"order_id": "888", "status": "已签收", "amount": 299, "remark": "7天内可申请退换"},
    "999": {"order_id": "999", "status": "运输中", "amount": 1599, "remark": "待签收"},
}

_VIP_POLICIES: Dict[str, dict] = {
    "普通用户": {"vip_level": "普通用户", "return_days": 7, "support": "基础客服"},
    "黄金VIP": {"vip_level": "黄金VIP", "return_days": 15, "support": "优先客服"},
    "白金VIP": {"vip_level": "白金VIP", "return_days": 30, "support": "专属客服"},
    "黑钻会员": {"vip_level": "黑钻会员", "return_days": 30, "support": "专属客服"},
    "SVIP": {"vip_level": "SVIP", "return_days": 30, "support": "专属客服"},
}


def get_user_profile(user_id: str) -> dict:
    return _USERS.get(user_id, {"user_id": user_id, "found": False})


def get_order_info(order_id: str) -> dict:
    return _ORDERS.get(order_id, {"order_id": order_id, "found": False})


def get_vip_policy(vip_level: str) -> dict:
    return _VIP_POLICIES.get(vip_level, {"vip_level": vip_level, "found": False})
