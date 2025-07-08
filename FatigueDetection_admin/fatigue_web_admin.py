#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
疲劳检测系统管理员Web界面
基于FastAPI的Web版本，提供疲劳记录查询和统计功能
"""

from fastapi import FastAPI, Request, Form, HTTPException, Depends, Query
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.security import HTTPBasic, HTTPBasicCredentials
import datetime
import csv
from typing import List, Dict, Optional
from io import StringIO
from starlette.responses import StreamingResponse
import secrets
import os
import pytz

# 导入数据库配置
from database_config import get_db_connection, init_database

app = FastAPI(title="疲劳检测系统 - 管理员界面")

# 创建必要的目录
os.makedirs("templates", exist_ok=True)
os.makedirs("static", exist_ok=True)

# 设置模板和静态文件
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

# 默认管理员账号密码
ADMIN_CREDENTIALS = {
    "admin": "admin123",
    "manager": "manager123",
    "root": "root123",
}

# 设置时区
TIMEZONE = pytz.timezone('Asia/Shanghai')  # 中国时区

# 会话管理（简单实现）
admin_sessions = {}

def get_current_admin(request: Request):
    """获取当前登录的管理员"""
    session_id = request.cookies.get("admin_session")
    if session_id and session_id in admin_sessions:
        return admin_sessions[session_id]
    return None

def require_admin_login(request: Request):
    """要求管理员登录"""
    admin = get_current_admin(request)
    if not admin:
        raise HTTPException(status_code=401, detail="需要管理员登录")
    return admin

@app.on_event("startup")
async def startup_event():
    """应用启动时初始化数据库"""
    try:
        init_database()
        print("数据库初始化成功")
    except Exception as e:
        print(f"数据库初始化失败: {e}")

@app.get("/", response_class=HTMLResponse)
async def admin_login_page(request: Request):
    """管理员登录页面"""
    return templates.TemplateResponse("admin_login.html", {"request": request})

@app.post("/login")
async def admin_login(request: Request, username: str = Form(...), password: str = Form(...)):
    """管理员登录处理"""
    if username in ADMIN_CREDENTIALS and ADMIN_CREDENTIALS[username] == password:
        # 创建会话，使用正确的时区
        current_time = datetime.datetime.now(TIMEZONE)
        session_id = f"admin_{username}_{current_time.timestamp()}"
        admin_sessions[session_id] = {
            "username": username,
            "login_time": current_time
        }
        
        # 重定向到管理界面
        response = RedirectResponse(url="/dashboard", status_code=302)
        response.set_cookie("admin_session", session_id, max_age=3600*8)  # 8小时有效
        return response
    else:
        return templates.TemplateResponse("admin_login.html", {
            "request": request,
            "error": "用户名或密码错误"
        })

@app.get("/dashboard", response_class=HTMLResponse)
async def admin_dashboard(request: Request, admin: dict = Depends(require_admin_login)):
    """管理员主界面"""
    return templates.TemplateResponse("admin_dashboard.html", {
        "request": request,
        "admin": admin
    })

@app.get("/api/fatigue-records")
async def get_fatigue_records(
    request: Request,
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    username: str = Query(""),
    fatigue_status: str = Query("全部"),
    start_time: str = Query(""),
    end_time: str = Query(""),
    admin: dict = Depends(require_admin_login)
):
    """获取疲劳记录API"""
    try:
        offset = (page - 1) * page_size
        
        with get_db_connection() as conn:
            cursor = conn.cursor()
            
            # 构建查询条件
            where_conditions = []
            params = []
            
            # 时间条件
            if start_time:
                where_conditions.append("timestamp >= %s")
                params.append(start_time)
            if end_time:
                where_conditions.append("timestamp <= %s") 
                params.append(end_time)
                
            # 用户名条件
            if username:
                where_conditions.append("username LIKE %s")
                params.append(f"%{username}%")
                
            # 疲劳状态条件
            if fatigue_status != "全部":
                where_conditions.append("fatigue_level = %s")
                params.append(fatigue_status)
            
            # 构建WHERE子句
            where_clause = ""
            if where_conditions:
                where_clause = "WHERE " + " AND ".join(where_conditions)
            
            # 查询总记录数
            count_query = f"SELECT COUNT(*) FROM fatigue_records {where_clause}"
            cursor.execute(count_query, params)
            total_records = cursor.fetchone()[0]
            
            # 查询分页数据
            query = f"""
                SELECT username, timestamp, fatigue_level
                FROM fatigue_records
                {where_clause}
                ORDER BY timestamp DESC
                LIMIT %s OFFSET %s
            """
            cursor.execute(query, params + [page_size, offset])
            records = cursor.fetchall()
            
        # 格式化记录
        formatted_records = []
        for record in records:
            formatted_records.append({
                "username": record[0],
                "timestamp": record[1].strftime("%Y-%m-%d %H:%M:%S") if record[1] else "",
                "fatigue_level": record[2]
            })
            
        return {
            "records": formatted_records,
            "total": total_records,
            "page": page,
            "page_size": page_size,
            "total_pages": (total_records + page_size - 1) // page_size
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"查询失败: {str(e)}")

@app.get("/api/statistics")
async def get_statistics(admin: dict = Depends(require_admin_login)):
    """获取统计信息API"""
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            
            # 总体统计
            cursor.execute("SELECT COUNT(*) FROM fatigue_records")
            total_records = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(DISTINCT username) FROM fatigue_records")
            total_users = cursor.fetchone()[0]
            
            cursor.execute("""
                SELECT fatigue_level, COUNT(*)
                FROM fatigue_records
                WHERE fatigue_level IN ('轻度疲劳', '中度疲劳', '重度疲劳')
                GROUP BY fatigue_level
            """)
            fatigue_stats = dict(cursor.fetchall())
            
            # 最近7天的记录
            week_ago = datetime.datetime.now() - datetime.timedelta(days=7)
            cursor.execute("""
                SELECT COUNT(*) FROM fatigue_records
                WHERE timestamp >= %s
            """, (week_ago,))
            recent_records = cursor.fetchone()[0]
            
            # 用户详细统计
            cursor.execute("""
                SELECT
                    username,
                    SUM(CASE WHEN fatigue_level = '轻度疲劳' THEN 1 ELSE 0 END) as mild_count,
                    SUM(CASE WHEN fatigue_level = '中度疲劳' THEN 1 ELSE 0 END) as moderate_count,
                    SUM(CASE WHEN fatigue_level = '重度疲劳' THEN 1 ELSE 0 END) as severe_count,
                    MAX(timestamp) as last_record
                FROM fatigue_records
                GROUP BY username
                ORDER BY (SUM(CASE WHEN fatigue_level = '轻度疲劳' THEN 1 ELSE 0 END) +
                         SUM(CASE WHEN fatigue_level = '中度疲劳' THEN 1 ELSE 0 END) +
                         SUM(CASE WHEN fatigue_level = '重度疲劳' THEN 1 ELSE 0 END)) DESC
            """)
            user_stats = cursor.fetchall()
            
        # 格式化用户统计
        formatted_user_stats = []
        for stats in user_stats:
            formatted_user_stats.append({
                "username": stats[0],
                "mild_count": stats[1],
                "moderate_count": stats[2], 
                "severe_count": stats[3],
                "last_record": stats[4].strftime("%Y-%m-%d %H:%M:%S") if stats[4] else ""
            })
            
        return {
            "total_records": total_records,
            "total_users": total_users,
            "recent_records": recent_records,
            "fatigue_stats": {
                "mild": fatigue_stats.get('轻度疲劳', 0),
                "moderate": fatigue_stats.get('中度疲劳', 0),
                "severe": fatigue_stats.get('重度疲劳', 0)
            },
            "user_stats": formatted_user_stats
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取统计信息失败: {str(e)}")

@app.get("/api/users")
async def get_users(admin: dict = Depends(require_admin_login)):
    """获取用户列表API"""
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT username, created_at
                FROM users
                ORDER BY created_at DESC
            """)
            users = cursor.fetchall()
            
        formatted_users = []
        for user in users:
            formatted_users.append({
                "username": user[0],
                "created_at": user[1].strftime("%Y-%m-%d %H:%M:%S") if user[1] else ""
            })
            
        return {"users": formatted_users}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取用户列表失败: {str(e)}")

@app.post("/api/users")
async def add_user(
    request: Request,
    username: str = Form(...),
    password: str = Form(...),
    admin: dict = Depends(require_admin_login)
):
    """添加用户API"""
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO users (username, password)
                VALUES (%s, %s)
            """, (username, password))
            conn.commit()
            
        return {"success": True, "message": "用户添加成功"}
        
    except Exception as e:
        if "Duplicate entry" in str(e):
            raise HTTPException(status_code=400, detail="用户名已存在")
        else:
            raise HTTPException(status_code=500, detail=f"添加用户失败: {str(e)}")

@app.get("/export-csv")
async def export_records_csv(
    username: str = Query(""),
    fatigue_status: str = Query("全部"),
    start_time: str = Query(""),
    end_time: str = Query(""),
    admin: dict = Depends(require_admin_login)
):
    """导出疲劳记录为CSV"""
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()

            # 构建查询条件（与获取记录相同的逻辑）
            where_conditions = []
            params = []

            if start_time:
                where_conditions.append("timestamp >= %s")
                params.append(start_time)
            if end_time:
                where_conditions.append("timestamp <= %s")
                params.append(end_time)
            if username:
                where_conditions.append("username LIKE %s")
                params.append(f"%{username}%")
            if fatigue_status != "全部":
                where_conditions.append("fatigue_level = %s")
                params.append(fatigue_status)

            where_clause = ""
            if where_conditions:
                where_clause = "WHERE " + " AND ".join(where_conditions)

            query = f"""
                SELECT username, timestamp, fatigue_level
                FROM fatigue_records
                {where_clause}
                ORDER BY timestamp DESC
            """
            cursor.execute(query, params)
            records = cursor.fetchall()

        # 生成CSV内容
        output = StringIO()
        # 写入BOM头，确保Excel正确识别UTF-8编码
        output.write('\ufeff')
        writer = csv.writer(output)
        writer.writerow(["用户名", "时间", "疲劳等级"])

        for record in records:
            formatted_record = [
                record[0],
                record[1].strftime("%Y-%m-%d %H:%M:%S") if record[1] else "",
                record[2]
            ]
            writer.writerow(formatted_record)

        csv_content = output.getvalue()
        output.close()

        # 返回CSV文件，使用正确的时区
        current_time = datetime.datetime.now(TIMEZONE) if 'TIMEZONE' in globals() else datetime.datetime.now()
        filename = f"fatigue_records_{current_time.strftime('%Y%m%d_%H%M%S')}.csv"

        def generate():
            yield csv_content.encode('utf-8')  # 使用UTF-8编码

        return StreamingResponse(
            generate(),
            media_type="text/csv; charset=utf-8",
            headers={
                "Content-Disposition": f"attachment; filename*=UTF-8''{filename}",
                "Content-Type": "text/csv; charset=utf-8"
            }
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"导出失败: {str(e)}")

@app.post("/logout")
async def admin_logout(request: Request):
    """管理员退出"""
    session_id = request.cookies.get("admin_session")
    if session_id and session_id in admin_sessions:
        del admin_sessions[session_id]

    response = RedirectResponse(url="/", status_code=302)
    response.delete_cookie("admin_session")
    return response

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8001, reload=True)
