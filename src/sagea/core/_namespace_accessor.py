#!/use/bin/env python
# coding=utf-8
# @Author  : Shuhao Liu
# @Time    : 2026/7/22 11:11 
# @File    : _namespace_accessor.py.py
# !/usr/bin/env python
# coding=utf-8

from __future__ import annotations

import inspect
import warnings
from dataclasses import dataclass
from functools import wraps
from typing import Any, Callable, Generic, Iterator, Literal, TypeVar, overload

NamespaceMethodKind = Literal["class", "instance"]

F = TypeVar("F", bound=Callable[..., Any])
A = TypeVar("A")


@dataclass(frozen=True)
class NamespaceMethodMeta:
    namespace: str
    summary: str | None = None
    order: int = 0
    kind: NamespaceMethodKind = "instance"
    show: bool = True


def namespace_method(
        *,
        namespace: str,
        summary: str | None = None,
        order: int = 0,
        kind: NamespaceMethodKind = "instance",
        show: bool = True,
) -> Callable[[F], F]:
    """
    Generic decorator for SHC namespace methods.

    Parameters
    ----------
    namespace:
        Namespace name, e.g. "filter", "io", "generate".
    summary:
        Short description shown in help().
        If None, the first line of docstring will be used.
    order:
        Display order.
    kind:
        "class":
            Method should be called as SHC.<namespace>.<method>(...).
        "instance":
            Method should be called as shc.<namespace>.<method>(...).
    show:
        Whether to show this method in help().
    """

    if kind not in {"class", "instance"}:
        raise ValueError(f"Invalid namespace method kind: {kind!r}")

    def decorator(func: F) -> F:
        meta = NamespaceMethodMeta(
            namespace=namespace,
            summary=summary,
            order=order,
            kind=kind,
            show=show,
        )

        @wraps(func)
        def wrapper(self, *args, **kwargs):
            is_class_context = self._obj is None
            owner_name = self._owner.__name__

            if kind == "instance" and is_class_context:
                raise TypeError(
                    f"`{func.__name__}` is an instance {namespace} method. "
                    f"Use `{owner_name.lower()}.{namespace}.{func.__name__}(...)`, not "
                    f"`{owner_name}.{namespace}.{func.__name__}(...)`."
                )

            if kind == "class" and not is_class_context:
                raise TypeError(
                    f"`{func.__name__}` is a class {namespace} method. "
                    f"Use `{owner_name}.{namespace}.{func.__name__}(...)`, not "
                    f"`{owner_name.lower()}.{namespace}.{func.__name__}(...)`."
                )

            return func(self, *args, **kwargs)

        wrapper.__namespace_method_meta__ = meta  # type: ignore[attr-defined]

        return wrapper  # type: ignore[return-value]

    return decorator


class NamespaceAccessorDescriptor(Generic[A]):
    """
    Descriptor supporting both class and instance namespace access.

    Examples
    --------
    SHC.io
    shc.io

    SHC.generate
    shc.generate

    SHC.filter
    shc.filter
    """

    def __init__(
            self,
            accessor_cls: type[A],
            *,
            cache_instance: bool = True,
            cache_class: bool = True,
    ) -> None:
        self.accessor_cls = accessor_cls
        self.cache_instance = cache_instance
        self.cache_class = cache_class
        self.name: str | None = None
        self.instance_cache_name: str | None = None
        self.class_cache_name: str | None = None

    def __set_name__(self, owner, name: str) -> None:
        self.name = name
        self.instance_cache_name = f"__cached_{name}_accessor__"
        self.class_cache_name = f"__cached_class_{name}_accessor__"

    @overload
    def __get__(self, obj: None, owner: type) -> A:
        ...

    @overload
    def __get__(self, obj: object, owner: type | None = None) -> A:
        ...

    def __get__(self, obj, owner=None):
        if owner is None:
            owner = type(obj)

        # Class access: SHC.xxx
        if obj is None:
            if self.cache_class and self.class_cache_name is not None:
                accessor = owner.__dict__.get(self.class_cache_name)
                if accessor is None:
                    accessor = self.accessor_cls(obj=None, owner=owner)
                    setattr(owner, self.class_cache_name, accessor)
                return accessor

            return self.accessor_cls(obj=None, owner=owner)

        # Instance access: shc.xxx
        if self.cache_instance and self.instance_cache_name is not None:
            if hasattr(obj, "__dict__"):
                cache = obj.__dict__
                accessor = cache.get(self.instance_cache_name)
                if accessor is None:
                    accessor = self.accessor_cls(obj=obj, owner=owner)
                    cache[self.instance_cache_name] = accessor
                return accessor

        return self.accessor_cls(obj=obj, owner=owner)


class BaseNamespaceAccessor:
    """
    Base class for SHC namespace accessors.

    Subclasses should define:

        _namespace_name = "io"
        _class_usage_name = "SHC.io"
        _instance_usage_name = "shc.io"

    and decorate public methods using namespace_method.
    """

    _namespace_name: str = "namespace"

    def __init__(self, obj: Any | None, owner: type) -> None:
        self._obj = obj
        self._owner = owner

    @property
    def _is_class_context(self) -> bool:
        return self._obj is None

    @property
    def _context_name(self) -> str:
        return "class" if self._is_class_context else "instance"

    @property
    def _owner_name(self) -> str:
        return self._owner.__name__

    @property
    def _class_usage_name(self) -> str:
        return f"{self._owner_name}.{self._namespace_name}"

    @property
    def _instance_usage_name(self) -> str:
        return f"{self._owner_name.lower()}.{self._namespace_name}"

    # ============================================================
    # Public display / help API
    # ============================================================

    def __str__(self) -> str:
        return self._format_methods(verbose=False)

    def __repr__(self) -> str:
        class_methods = [
            name for name, _, _ in self._iter_namespace_methods(kind="class")
        ]
        instance_methods = [
            name for name, _, _ in self._iter_namespace_methods(kind="instance")
        ]

        return (
            f"<{type(self).__name__} "
            f"context={self._context_name!r} "
            f"class_methods={class_methods} "
            f"instance_methods={instance_methods}>"
        )

    def __help__(self) -> str:
        """
        Non-standard helper.

        Python built-in help(obj) does not automatically call __help__().
        Prefer:

            print(SHC.xxx.help())
            print(shc.xxx.help())
        """
        return self._format_methods(verbose=True)

    def help(self) -> str:
        return self.__help__()

    def __dir__(self) -> list[str]:
        default_attrs = set(super().__dir__())
        method_names = {
            name for name, _, _ in self._iter_namespace_methods(kind=None)
        }
        return sorted(default_attrs | method_names)

    # ============================================================
    # Deprecated dispatch style
    # ============================================================

    def __call__(self, name: str | None = None, *args, **kwargs):
        """
        Deprecated usage:

            shc.io("method", ...)
            shc.filter("method", ...)

        Recommended:

            shc.io.method(...)
            shc.filter.method(...)
        """
        if name is None:
            raise TypeError(
                f"`{self._class_usage_name}` is a namespace, not a standalone callable. "
                f"Use `{self._class_usage_name}.<method_name>(...)` or "
                f"`{self._instance_usage_name}.<method_name>(...)`."
            )

        warnings.warn(
            f"`{self._namespace_name}(name, ...)` is deprecated. "
            f"Use `{self._namespace_name}.name(...)` instead.",
            DeprecationWarning,
            stacklevel=2,
        )

        method = getattr(self, name)
        return method(*args, **kwargs)

    # ============================================================
    # Method discovery
    # ============================================================

    @classmethod
    def _registered_namespace_methods(
            cls,
    ) -> dict[str, tuple[Callable[..., Any], NamespaceMethodMeta]]:
        """
        Collect decorated namespace methods from MRO.

        Subclass methods override base class methods.
        """
        methods: dict[str, tuple[Callable[..., Any], NamespaceMethodMeta]] = {}

        for base in reversed(cls.mro()):
            for name, obj in base.__dict__.items():
                if isinstance(obj, (classmethod, staticmethod)):
                    func = obj.__func__
                else:
                    func = obj

                meta = getattr(func, "__namespace_method_meta__", None)

                if not isinstance(meta, NamespaceMethodMeta):
                    continue

                if not meta.show:
                    continue

                if meta.namespace != cls._namespace_name:
                    continue

                methods[name] = (func, meta)

        return methods

    @classmethod
    def _iter_namespace_methods(
            cls,
            *,
            kind: NamespaceMethodKind | None = None,
    ) -> Iterator[tuple[str, Callable[..., Any], NamespaceMethodMeta]]:
        items = []

        for name, (func, meta) in cls._registered_namespace_methods().items():
            if kind is not None and meta.kind != kind:
                continue
            items.append((meta.order, name, func, meta))

        for _, name, func, meta in sorted(items, key=lambda x: (x[0], x[1])):
            yield name, func, meta

    # ============================================================
    # Formatting
    # ============================================================

    @staticmethod
    def _method_signature(func: Callable[..., Any]) -> str:
        try:
            sig = inspect.signature(func)
        except ValueError:
            return "(...)"

        params = list(sig.parameters.values())

        if params and params[0].name in {"self", "cls"}:
            params = params[1:]

        sig = sig.replace(parameters=params)
        return str(sig)

    @staticmethod
    def _method_summary(
            func: Callable[..., Any],
            meta: NamespaceMethodMeta,
    ) -> str:
        if meta.summary:
            return meta.summary

        doc = inspect.getdoc(func)
        if not doc:
            return ""

        return doc.strip().splitlines()[0]

    def _format_methods(self, *, verbose: bool) -> str:
        class_methods = list(self._iter_namespace_methods(kind="class"))
        instance_methods = list(self._iter_namespace_methods(kind="instance"))

        lines: list[str] = [
            f"{self._owner_name}.{self._namespace_name} accessor",
            f"Current context: {self._context_name}",
            "",
            "Below are class methods:",
            f"  Recommended usage: {self._class_usage_name}.<method_name>(...)",
        ]

        if class_methods:
            for name, func, meta in class_methods:
                lines.extend(
                    self._format_one_method(
                        name=name,
                        func=func,
                        meta=meta,
                        verbose=verbose,
                    )
                )
        else:
            lines.append("  <none>")

        lines.extend(
            [
                "",
                "Below are instance methods:",
                f"  Recommended usage: {self._instance_usage_name}.<method_name>(...)",
            ]
        )

        if instance_methods:
            for name, func, meta in instance_methods:
                lines.extend(
                    self._format_one_method(
                        name=name,
                        func=func,
                        meta=meta,
                        verbose=verbose,
                    )
                )
        else:
            lines.append("  <none>")

        return "\n".join(lines)

    def _format_one_method(
            self,
            *,
            name: str,
            func: Callable[..., Any],
            meta: NamespaceMethodMeta,
            verbose: bool,
    ) -> list[str]:
        signature = self._method_signature(func)
        summary = self._method_summary(func, meta)

        if meta.kind == "class":
            full_name = f"{self._class_usage_name}.{name}"
        else:
            full_name = f"{self._instance_usage_name}.{name}"

        if verbose:
            lines = [f"  - {full_name}{signature}"]
            if summary:
                lines.append(f"      {summary}")

            doc = inspect.getdoc(func)
            if doc:
                lines.append("      Doc:")
                for line in doc.splitlines():
                    lines.append(f"        {line}")

            return lines

        if summary:
            return [f"  - {name}{signature}: {summary}"]

        return [f"  - {name}{signature}"]
