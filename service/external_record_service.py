#外部使用记录数据服务（数据库版）
#【重构】原 CSV 解析 + 内存缓存升级为 SQLite 查询：建表、迁移与增删改查下沉到
#service/database_service.py，本类只负责按 (用户ID, 购买时间) 组装对外查询结果。
#对外接口保持不变（get_records/get_user_months/get_warranty_info/reload/_load_error），
#上层 agent/tools/agent_tools.py 无需任何改动。注意：返回给模型的数据不含密码字段。
from datetime import date
from typing import Any

from service.database_service import get_database_service
from utils.config_handler import agent_config
from utils.logger_handler import logger


class ExternalRecordService:
    """外部使用记录数据服务（数据库版）

    - 数据源为 SQLite（用户/购买/维修三张表），查询时实时组装，无需内存缓存
    - _load_error 区分两种失败：数据源故障（数据库不可用）与业务上无记录
    """

    def __init__(self, db_service=None) -> None:
        #数据库服务可注入（测试用），默认取全局单例
        self._db = db_service or get_database_service()
        self._load_error: str | None = None   #数据源故障时置非空；None 表示数据源正常

    def _ensure_ready(self) -> bool:
        """检查数据源是否可用；故障时置 _load_error，由上层区分"数据源故障"与"无记录" """
        if self._db.error is not None:
            self._load_error = "外部数据数据库不可用"
            return False
        return True

    def get_records(self, user_id: str, month: str) -> dict[str, Any] | None:
        """查询指定用户在某月的购买记录及其维修记录；无记录返回 None，不抛异常。

        返回结构（与原 CSV 版一致）：
        {"特征": [...], "外设类型": "...", "售后记录": ["维修日期:损坏原因", ...], "购买时间": "..."}
        密码只存于 users 表、仅用于登录校验，不在此返回。
        """
        if not self._ensure_ready():
            return None
        purchase = self._db.get_purchase(user_id, month)
        if purchase is None:
            return None
        #维修记录逐条组装回"维修日期:损坏原因"格式（与原 CSV 字段语义一致）
        repairs = [f"{r['维修日期']}:{r['损坏原因']}" for r in self._db.get_repairs(purchase["购买ID"])]
        feature = purchase["特征"]
        return {
            "特征": [s.strip() for s in feature.split("|") if s.strip()],
            "外设类型": purchase["外设类型"],
            "售后记录": repairs,
            "购买时间": month,
        }

    def reload(self) -> None:
        """兼容旧接口：数据库查询无内存缓存，这里只清理数据源错误标记"""
        self._load_error = None

    def get_user_months(self, user_id: str) -> list[str]:
        """返回某用户全部购买时间（YYYY-MM）列表，升序；无记录返回空列表"""
        if not self._ensure_ready():
            return []
        months = [p["购买时间"] for p in self._db.get_purchases_by_user(user_id)]
        return sorted(months)

    #保修期计算：保修自购买时间起 warranty_months 个月（agent.yml 配置，默认12）
    def get_warranty_info(self, user_id: str, purchase_month: str) -> dict[str, Any] | None:
        """查询某用户某购买记录（按购买时间定位）的保修状态。

        返回 {"购买时间","保修截止","状态","剩余月数"或"已过期月数"}；
        无该购买记录返回 None；月份粒度比较，保修覆盖购买月起 warranty_months 个月。
        """
        if not self._ensure_ready():
            return None
        purchase = self._db.get_purchase(user_id, purchase_month)
        if purchase is None:
            return None

        months = int(agent_config.get("warranty_months", 12))
        year, month = map(int, purchase_month.split("-"))
        #把年月折算成"总月数"便于加减：购买月为第0个月，覆盖到第 months-1 个月，第 months 个月起过保
        expiry_total = year * 12 + (month - 1) + months
        exp_year, exp_rem = divmod(expiry_total, 12)
        expiry = f"{exp_year:04d}-{exp_rem + 1:02d}"

        now = date.today()
        now_total = now.year * 12 + now.month - 1
        info: dict[str, Any] = {"购买时间": purchase_month, "保修截止": expiry}

        if now_total >= expiry_total:
            info["状态"] = "已过保修期"
            info["已过期月数"] = now_total - expiry_total + 1
        else:
            info["状态"] = "保修期内"
            info["剩余月数"] = expiry_total - now_total
        return info


#模块级惰性单例：导入本模块零开销，首次调用才连接数据库
_instance: ExternalRecordService | None = None


def get_external_record_service() -> ExternalRecordService:
    """获取全局唯一的数据服务实例（惰性创建）"""
    global _instance
    if _instance is None:
        _instance = ExternalRecordService()
    return _instance


if __name__ == "__main__":
    #运行方式：cd 项目根 && .venv/Scripts/python.exe -m service.external_record_service
    svc = get_external_record_service()
    print("存在记录:", svc.get_records("2483", "2025-04"))
    print("不存在记录:", svc.get_records("2483", "1999-01"))
    print("多笔购买用户4671的月份:", svc.get_user_months("4671"))
    print("保修(4671/2025-01):", svc.get_warranty_info("4671", "2025-01"))
