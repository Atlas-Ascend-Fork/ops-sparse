/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef SPSV_ARCH22_SPSV_TORCH_COMPAT_H_
#define SPSV_ARCH22_SPSV_TORCH_COMPAT_H_
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <string>
#include "acl/acl_rt.h"

struct _object;
struct _ts;

namespace spsv_torch_compat {
enum class Preparation { Ready, ProbeFailure, SubmissionFailure };
namespace python {
using Object = ::_object;
using RuntimeProbe = int (*)();

template<class T> inline T Symbol(const char *name)
{
    return reinterpret_cast<T>(dlsym(RTLD_DEFAULT, name));
}

inline RuntimeProbe FinalizationProbe()
{
    auto probe = Symbol<RuntimeProbe>("Py_IsFinalizing");
    return probe != nullptr ? probe : Symbol<RuntimeProbe>("_Py_IsFinalizing");
}

struct Api {
    int (*initialized)() = Symbol<int (*)()>("Py_IsInitialized");
    RuntimeProbe finalizing = FinalizationProbe();
    int (*holdsGil)() = Symbol<int (*)()>("PyGILState_Check");
    int (*ensure)() = Symbol<int (*)()>("PyGILState_Ensure");
    void (*release)(int) = Symbol<void (*)(int)>("PyGILState_Release");
    ::_ts *(*threadState)() = Symbol<::_ts *(*)()>("PyGILState_GetThisThreadState");
    Object *(*modules)() = Symbol<Object *(*)()>("PyImport_GetModuleDict");
    Object *(*lookup)(Object *, const char *) =
        Symbol<Object *(*)(Object *, const char *)>("PyDict_GetItemString");
    Object *(*attribute)(Object *, const char *) =
        Symbol<Object *(*)(Object *, const char *)>("PyObject_GetAttrString");
    Object *(*integer)(long) = Symbol<Object *(*)(long)>("PyLong_FromLong");
    Object *(*call)(Object *, ...) = Symbol<Object *(*)(Object *, ...)>("PyObject_CallFunctionObjArgs");
    void *(*pointer)(Object *) = Symbol<void *(*)(Object *)>("PyLong_AsVoidPtr");
    void (*decref)(Object *) = Symbol<void (*)(Object *)>("Py_DecRef");
    Object *(*error)() = Symbol<Object *(*)()>("PyErr_Occurred");
    void (*fetch)(Object **, Object **, Object **) =
        Symbol<void (*)(Object **, Object **, Object **)>("PyErr_Fetch");
    void (*restore)(Object *, Object *, Object *) =
        Symbol<void (*)(Object *, Object *, Object *)>("PyErr_Restore");
    void (*clear)() = Symbol<void (*)()>("PyErr_Clear");
    int (*matches)(Object*) = Symbol<int (*)(Object*)>("PyErr_ExceptionMatches");
    Object** runtimeError = Symbol<Object**>("PyExc_RuntimeError");

    bool hasThreadApi() const { return initialized && finalizing && holdsGil && ensure && release && threadState; }

    bool hasObjectApi() const
    {
        return modules && lookup && attribute && integer && call && pointer && decref && error && fetch && restore &&
               clear && matches && runtimeError;
    }

    bool complete() const { return hasThreadApi() && hasObjectApi(); }
};

class Reference {
public:
    Reference(const Api &api, Object *object) : api_(api), object_(object) {}
    ~Reference()
    {
        if (object_ != nullptr)
            api_.decref(object_);
    }
    Reference(const Reference &) = delete;
    Reference &operator=(const Reference &) = delete;
    Object *get() const { return object_; }
private:
    const Api &api_;
    Object *object_;
};

class Context {
public:
    explicit Context(const Api& api) : api_(api), gil_(api.holdsGil() ? -1 : api.ensure())
    {
        api_.fetch(&type_, &value_, &traceback_);
    }
    ~Context()
    {
        // Clear only this adapter's errors, preserving the exception that
        // belonged to the caller before entering the library.
        api_.clear();
        api_.restore(type_, value_, traceback_);
        if (gil_ != -1)
            api_.release(gil_);
    }
    Context(const Context &) = delete;
    Context &operator=(const Context &) = delete;
private:
    const Api &api_;
    int gil_;
    Object *type_ = nullptr;
    Object *value_ = nullptr;
    Object *traceback_ = nullptr;
};

inline Preparation Submit(const Api& api, Object* stream)
{
    Reference raw(api, api.attribute(stream, "npu_stream"));
    if (!raw.get()) {
        return api.matches(*api.runtimeError) ? Preparation::SubmissionFailure : Preparation::ProbeFailure;
    }
    void* actual = api.pointer(raw.get());
    if (api.error() != nullptr || actual == nullptr)
        return Preparation::ProbeFailure;
    return Preparation::Ready;
}

inline Preparation CurrentStream(const Api& api, Object* npu, int device)
{
    Reference current(api, api.attribute(npu, "current_stream"));
    Reference index(api, api.integer(device));
    if (!current.get() || !index.get())
        return Preparation::ProbeFailure;
    Reference stream(api, api.call(current.get(), index.get(), static_cast<Object*>(nullptr)));
    return stream.get() ? Submit(api, stream.get()) : Preparation::ProbeFailure;
}

inline Preparation Prepare(int device, const Api& api)
{
    if (api.initialized == nullptr || !api.initialized())
        return Preparation::Ready;
    if (!api.complete() || api.finalizing())
        return Preparation::ProbeFailure;
    // A framework queue worker must not wait for its own queue. Native threads
    // without Python state retain their normal caller-managed ACL ordering.
    if (api.threadState() == nullptr)
        return Preparation::Ready;
    static thread_local bool active = false;
    if (active)
        return Preparation::SubmissionFailure;
    struct Guard {
        bool &active;
        explicit Guard(bool &flag) : active(flag) { active = true; }
        ~Guard() { active = false; }
    } guard(active);
    Context context(api);
    Object *modules = api.modules();
    if (modules == nullptr)
        return Preparation::ProbeFailure;
    Object *npu = api.lookup(modules, "torch_npu.npu");
    if (npu == nullptr)
        return api.error() ? Preparation::ProbeFailure : Preparation::Ready;
    return CurrentStream(api, npu, device);
}

inline Preparation Prepare(int device)
{
    // Resolve the C API once; the module lookup still permits late torch_npu
    // imports into an already initialized interpreter. Do not initialize Python.
    static const Api api;
    return Prepare(device, api);
}
} // namespace python
} // namespace spsv_torch_compat

namespace spsv_torch_compat {
using PrepareFunction = int (*)(int32_t, void *);
using WarningFunction = void (*)(const char*);

struct Module {
    std::string path;
    void *handle = nullptr;
    PrepareFunction prepare = nullptr;
    explicit Module(const char *name) : path(name)
    {
        handle = dlopen(name, RTLD_NOW | RTLD_LOCAL);
        if (handle != nullptr) {
            prepare = reinterpret_cast<PrepareFunction>(
                dlsym(handle, "OpsSparseSpSVTorchQueuePrepareV1"));
        }
        // Retain the module for process lifetime: concurrent calls may use it.
    }
};

inline void Warn(WarningFunction warning, const char* message)
{
    if (warning != nullptr)
        warning(message);
}

inline bool Prepare(aclrtStream stream, WarningFunction warning = nullptr)
{
    const char* enabled = std::getenv("OPS_SPARSE_SPSV_TORCH_QUEUE");
    if (enabled != nullptr && std::strcmp(enabled, "0") == 0)
        return true;
    int32_t device = -1;
    if (aclrtGetDevice(&device) != ACL_SUCCESS) return false;
    const char* path = std::getenv("OPS_SPARSE_SPSV_TORCH_QUEUE_LIBRARY");
    if (path != nullptr && *path != '\0') {
        if (*path == '/') {
            // Retain the first module for process lifetime. Invalid overrides
            // fall back to automatic discovery instead of rejecting ACL work.
            static Module module(path);
            if (module.path == path && module.prepare != nullptr) {
                return module.prepare(device, stream) == 0;
            }
        }
        Warn(warning, "Framework queue module unavailable; using automatic discovery.");
    }
    Preparation result = python::Prepare(device);
    if (result == Preparation::ProbeFailure) {
        Warn(warning, "Framework queue discovery unavailable; preserving caller-managed ACL stream ordering.");
    }
    return result != Preparation::SubmissionFailure;
}
}

#endif // SPSV_ARCH22_SPSV_TORCH_COMPAT_H_
