"""业务数据库初始化脚本。"""

from __future__ import annotations

from src.services.business_db import get_order_info, get_user_profile, get_vip_policy


if __name__ == "__main__":
    print(get_user_profile("U1001"))
    print(get_order_info("888"))
    print(get_vip_policy("黄金VIP"))
