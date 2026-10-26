#pragma once

struct Vec2 {
    int y;
    int x;

    Vec2 operator+(const Vec2& other) const
    {
        return {y + other.y, x + other.x};
    }

    bool operator==(const Vec2& other) const
    {
        return y == other.y && x == other.x;
    }
};

inline constexpr Vec2 UP    = {-1, 0};
inline constexpr Vec2 DOWN  = {1, 0};
inline constexpr Vec2 LEFT  = {0, -1};
inline constexpr Vec2 RIGHT = {0, 1};