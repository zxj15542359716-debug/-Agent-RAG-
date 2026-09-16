#登录凭证服务（JWT）
#【新增】修复越权漏洞（IDOR）：原实现中 /api/chat、/api/report 直接信任请求体里的
# user_id 字段——任何人伪造该字段即可读取/操作他人数据，无需密码。
#现在：登录成功由服务端签发带签名的 JWT；业务接口通过 get_current_user 依赖
#从 Authorization: Bearer <token> 解析身份，不再信任请求体。
import jwt
from datetime import datetime, timedelta, timezone

from fastapi import Header, HTTPException

from utils.config_handler import require_env

#签名密钥从环境变量读取；缺失时导入即抛错（fail-fast：启动阶段暴露配置问题，
#而不是服务照常启动、用户请求时才失败）。密钥泄露=任何人可伪造任意用户身份。
_JWT_SECRET = require_env("JWT_SECRET")
_ALGO = "HS256"
_EXPIRE_HOURS = 2   #凭证有效期（小时）


def create_token(user_id: str) -> str:
    """为登录成功的用户签发 JWT：sub=用户ID，2 小时后过期"""
    now = datetime.now(timezone.utc)
    return jwt.encode(
        {"sub": user_id, "iat": now, "exp": now + timedelta(hours=_EXPIRE_HOURS)},
        _JWT_SECRET, algorithm=_ALGO,
    )


def get_current_user(authorization: str = Header(default="")) -> str:
    """FastAPI 依赖：从 Authorization: Bearer <token> 解析登录用户ID。

    校验失败一律抛 401（前端据此引导重新登录）；用户身份不再来自请求体，
    越权查询他人数据的路径被彻底关闭。
    """
    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="未登录")
    token = authorization[7:].strip()
    try:
        payload = jwt.decode(token, _JWT_SECRET, algorithms=[_ALGO])
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="登录已过期，请重新登录")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="无效的登录凭证") from None
    return payload["sub"]
