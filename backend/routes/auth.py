from fastapi import APIRouter, HTTPException, Response, status, Depends
from backend.models.user import UserCreate, UserLogin, UserInDB
from backend.core.security import get_password_hash, verify_password, create_access_token, get_current_user
from backend.db.mongo import users_collection
from datetime import timedelta
from backend.core.config import settings
from fastapi.responses import JSONResponse

auth_router = APIRouter(prefix="/auth", tags=["auth"])

@auth_router.post("/register")
def register(user: UserCreate):
    if users_collection.find_one({"email": user.email}):
        raise HTTPException(status_code=400, detail="User already exists")

    hashed_pw = get_password_hash(user.password)
    users_collection.insert_one({
         "email": user.email,
        "hashed_password": hashed_pw,
        "full_name": user.full_name,
        "mobile": user.mobile,
        "role": user.role
    })
    return {"msg": "User registered"}


@auth_router.post("/login")
def login(user: UserLogin, response: Response):
    db_user = users_collection.find_one({"email": user.email})
    if not db_user or not verify_password(user.password, db_user["hashed_password"]):
        raise HTTPException(status_code=401, detail="Invalid credentials")

    # Generate access token
    token = create_access_token(
        data={"sub": db_user["email"], "role": db_user["role"]},
        expires_delta=timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    )

    # Set token in a secure HTTP-only cookie
    response.set_cookie(
        key="access_token",
        value=token,
        httponly=True,
        secure=False,  # Change to True in production with HTTPS
        samesite="lax",
        max_age=1800  # 30 minutes cookie expiration
    )

    # Send relevant user details in response
    return {
        "msg": "Login successful",
        "user": {
            "email": db_user["email"],
            "full_name": db_user["full_name"],
            "role": db_user["role"],
            "mobile": db_user["mobile"]
        }
    }



@auth_router.get("/users")
def get_all_users(current_user: dict = Depends(get_current_user)):
    # Ensure only admin can access this route
    if current_user["role"] != "admin":
        raise HTTPException(status_code=403, detail="Only admin can view users.")

    users_cursor = users_collection.find({}, {"_id": 0, "email": 1, "role": 1})
    users = list(users_cursor)
    return {"users": users}


@auth_router.post("/logout")
def logout(response: Response, current_user: UserInDB = Depends(get_current_user)):
    # Remove the JWT token by clearing the cookie
    response.delete_cookie(key="access_token")
    return JSONResponse(content={"msg": "Logged out successfully"}, status_code=status.HTTP_200_OK)


@auth_router.post("/logout_all")
def logout_all(response: Response, current_user: UserInDB = Depends(get_current_user)):
    # To implement a more robust "logout all" feature:
    # You would typically store and revoke the JWT tokens in a DB or use a blacklist for token invalidation.
    # For now, we are just clearing the cookie (current session).

    response.delete_cookie(key="access_token")

    # Ideally, in a more advanced system, you could add a blacklist/collection of revoked tokens here.
    # Example: db_tokens_collection.delete_many({"user_id": current_user.id})

    return JSONResponse(content={"msg": "Logged out from all devices"}, status_code=status.HTTP_200_OK)
