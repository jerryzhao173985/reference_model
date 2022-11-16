
// Copyright (c) 2022, ARM Limited.
//
//    Licensed under the Apache License, Version 2.0 (the "License");
//    you may not use this file except in compliance with the License.
//    You may obtain a copy of the License at
//
//         http://www.apache.org/licenses/LICENSE-2.0
//
//    Unless required by applicable law or agreed to in writing, software
//    distributed under the License is distributed on an "AS IS" BASIS,
//    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//    See the License for the specific language governing permissions and
//    limitations under the License.

#ifndef ARRAY_PROXY_H_
#define ARRAY_PROXY_H_

#include <cstddef>
#include <type_traits>

template <typename T>
class ArrayProxy
{
public:
    ArrayProxy(size_t n, T* ptr) noexcept
        : _n(n)
        , _ptr(ptr)
    {}

    template <typename U = T, std::enable_if_t<std::is_const<U>::value, int> = 0>
    ArrayProxy(size_t n, typename std::remove_const_t<T>* ptr) noexcept
        : _n(n)
        , _ptr(ptr)
    {}

    template <std::size_t S>
    ArrayProxy(T (&ptr)[S]) noexcept
        : _n(S)
        , _ptr(ptr)
    {}

    template <std::size_t S, typename U = T, std::enable_if_t<std::is_const<U>::value, int> = 0>
    ArrayProxy(typename std::remove_const_t<T> (&ptr)[S]) noexcept
        : _n(S)
        , _ptr(ptr)
    {}

    template <typename O,
              std::enable_if_t<std::is_convertible_v<decltype(std::declval<O>().data()), T*> &&
                                   std::is_convertible_v<decltype(std::declval<O>().size()), std::size_t>,
                               int> = 0>
    ArrayProxy(O& obj) noexcept
        : _n(obj.size())
        , _ptr(obj.data())
    {}

    size_t size() const noexcept
    {
        return _n;
    }

    T* data() const noexcept
    {
        return _ptr;
    }

    bool empty() const noexcept
    {
        return _n == 0;
    }

    const T* begin() const noexcept
    {
        return _ptr;
    }

    const T* end() const noexcept
    {
        return _ptr + _n;
    }

    T& operator[](size_t idx) noexcept
    {
        return *(_ptr + idx);
    }

    const T& operator[](size_t idx) const noexcept
    {
        return *(_ptr + idx);
    }

private:
    size_t _n;
    T* _ptr;
};

#endif    // ARRAY_PROXY_H_
