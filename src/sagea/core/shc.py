#!/use/bin/env python
# coding=utf-8
# @Author  : Shuhao Liu
# @Time    : 2026/6/17 12:45 
# @File    : shc.py

# src/sagea/core/shc.py

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence
import datetime as dt
from functools import cached_property

import numpy as np

from core._shc_generator_wrapper import _SHCGeneratorAccessor
from core._shc_filter_wrapper import _SHCFilterAccessor

from sagea.constants.constant import PhysicalDimension
from sagea.utils import MathTool


class SHCDeprecationWarning(FutureWarning):
    pass


class _SHCMeta(type):
    @property
    def generate(cls) -> "_SHCGeneratorAccessor":
        accessor = cls.__dict__.get("_generate_accessor", None)
        if accessor is None:
            accessor = _SHCGeneratorAccessor(cls)
            setattr(cls, "_generate_accessor", accessor)
        return accessor

    def __dir__(cls) -> list[str]:
        names = set(super().__dir__())
        names.add("generate")
        return sorted(names)


@dataclass
class SHC(metaclass=_SHCMeta):
    """
    Spherical harmonic coefficients.

    Parameters
    ----------
    _values : np.ndarray
        1D or 2D SH coefficient array.
        Shape:
            - (ncoef,)
            - (ntime, ncoef)
        where ncoef = (lmax + 1) ** 2.
    normalization : str
        Normalization convention. Currently support "4pi".
    dates : sequence of datetime.date, optional
        Time stamps for each SHC epoch.
    attrs : dict
        Extra metadata.
    """

    _values: np.ndarray
    normalization: str = "4pi"
    dates: Sequence[dt.date] | None = None
    attrs: dict = field(default_factory=dict)

    def __post_init__(self):
        values = np.asarray(self._values, dtype=float)

        if values.ndim == 1:
            values = values[None, :]

        if values.ndim != 2:
            raise ValueError(
                f"SHC.values should be 1D or 2D, got shape {values.shape}."
            )

        lmax_float = np.sqrt(values.shape[1]) - 1
        lmax = int(round(lmax_float))

        if not np.isclose(lmax_float, lmax, atol=1e-8):
            raise ValueError(
                f"Invalid SHC coefficient length: {values.shape[1]}. "
                f"Expected (lmax + 1)^2."
            )

        if self.normalization not in ("4pi",):
            raise ValueError(f"Unsupported normalization: {self.normalization}")

        if self.dates is not None and len(self.dates) != values.shape[0]:
            raise ValueError(
                f"dates length {len(self.dates)} does not match "
                f"SHC time dimension {values.shape[0]}."
            )

        self._values = values

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------
    @classmethod
    def from_gfc(
            cls,
            filepath: str | Path | Sequence[str | Path],
            lmax: int,
            key: str = "gfc",
            cols=None,
            normalization: str = "4pi",
            dates: Sequence[dt.date] | None = None,
            attrs: dict | None = None,
    ) -> "SHC":
        """
        Read SHC from one or multiple GFC files.
        """
        from sagea.sgio.gfc_reader import read_gfc

        warnings.warn(
            (
                "`SHC.from_gfc(...)` is deprecated and will be removed in a "
                "future version. Use `SHC.generate.from_gfc(...)` instead. "
                "See README or documentation for migration details."
            ),
            category=SHCDeprecationWarning,
            stacklevel=2,
        )

        if isinstance(filepath, (str, Path)):
            cs = read_gfc(
                Path(filepath),
                key=key,
                lmax=lmax,
                col_indices=cols,
            )
        else:
            cs = np.asarray(
                [
                    read_gfc(
                        Path(path),
                        key=key,
                        lmax=lmax,
                        col_indices=cols,
                    )
                    for path in filepath
                ]
            )

        return cls(
            _values=cs,
            normalization=normalization,
            dates=dates,
            attrs={} if attrs is None else attrs,
        )

    @classmethod
    def from_trend(
            cls,
            shc_trend: "SHC",
            dates: Sequence[dt.date],
            ref_time: dt.date | None = None,
    ) -> "SHC":
        """
        Generate SHC time series from a trend SHC.

        C(t) = C_rate * (t - t_ref)
        """
        from sagea.utils import TimeTool

        warnings.warn(
            (
                "`SHC.from_trend(...)` is deprecated and will be removed in a "
                "future version. Use `SHC.generate.from_trend(...)` instead. "
                "See README or documentation for migration details."
            ),
            category=SHCDeprecationWarning,
            stacklevel=2,
        )

        if len(shc_trend) != 1:
            raise ValueError("shc_trend should contain only one epoch.")

        if ref_time is None:
            ref_time = dates[0]

        year_frac = TimeTool.convert_date_format(
            dates,
            input_type=TimeTool.DateFormat.ClassDate,
            output_type=TimeTool.DateFormat.YearFraction,
        )
        year_frac = np.asarray(year_frac)

        ref_year_frac = TimeTool.convert_date_format(
            ref_time,
            input_type=TimeTool.DateFormat.ClassDate,
            output_type=TimeTool.DateFormat.YearFraction,
        )

        dt_year = year_frac - ref_year_frac
        trend = shc_trend._values[0]

        values = dt_year[:, None] @ trend[None, :]

        return cls(
            _values=values,
            normalization=shc_trend.normalization,
            dates=dates,
            attrs=shc_trend.attrs.copy(),
        )

    generate: _SHCGeneratorAccessor

    def __getattribute__(self, name):
        if name == "generate":
            raise AttributeError(
                "`generate` can only be accessed from the SHC class. "
                "Use `SHC.generate.<method_name>(...)` instead."
            )
        return super().__getattribute__(name)

    # ------------------------------------------------------------------
    # Basic properties
    # ------------------------------------------------------------------
    @property
    def value(self) -> np.ndarray:
        """
        Backward-compatible alias.
        """
        return self._values

    @value.setter
    def value(self, arr: np.ndarray):
        self._values = np.asarray(arr, dtype=float)

    @property
    def lmax(self) -> int:
        return int(round(np.sqrt(self._values.shape[1]) - 1))

    @property
    def ntime(self) -> int:
        return self._values.shape[0]

    def __len__(self) -> int:
        return self.ntime

    def copy(self) -> "SHC":
        return SHC(
            _values=self._values.copy(),
            normalization=self.normalization,
            dates=None if self.dates is None else list(self.dates),
            attrs=self.attrs.copy(),
        )

    # ------------------------------------------------------------------
    # Representation conversion
    # ------------------------------------------------------------------
    @property
    def cs2d(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Return C/S coefficients as 3D arrays.

        Returns
        -------
        cqlm, sqlm : np.ndarray
            Shape: (ntime, lmax + 1, lmax + 1)
        """
        c_list = []
        s_list = []

        for cs in self._values:
            c, s = MathTool.cs_decompose_triangle1d_to_cs2d(cs)
            c_list.append(c)
            s_list.append(s)

        return np.asarray(c_list), np.asarray(s_list)

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------
    @property
    def mean(self) -> np.ndarray:
        return np.mean(self._values, axis=0)

    @property
    def std(self) -> np.ndarray:
        return np.std(self._values, axis=0)

    @property
    def covariance(self) -> np.ndarray:
        return np.cov(self._values.T)

    @property
    def degree_rms(self) -> np.ndarray:
        cqlm, sqlm = self.cs2d
        return MathTool.get_degree_rms(cqlm, sqlm)

    @property
    def degree_rss(self) -> np.ndarray:
        cqlm, sqlm = self.cs2d
        return MathTool.get_degree_rss(cqlm, sqlm)

    @property
    def cumulative_degree_rss(self) -> np.ndarray:
        cqlm, sqlm = self.cs2d
        return MathTool.get_cumulative_rss(cqlm, sqlm)

    # ------------------------------------------------------------------
    # Basic operations
    # ------------------------------------------------------------------
    def __add__(self, other: "SHC") -> "SHC":
        if not isinstance(other, SHC):
            raise TypeError("Can only add SHC with SHC.")

        if self.lmax != other.lmax:
            raise ValueError("lmax does not match.")

        return SHC(
            _values=self._values + other._values,
            normalization=self.normalization,
            dates=self.dates,
            attrs=self.attrs.copy(),
        )

    def __sub__(self, other: "SHC") -> "SHC":
        if not isinstance(other, SHC):
            raise TypeError("Can only subtract SHC with SHC.")

        if self.lmax != other.lmax:
            raise ValueError("lmax does not match.")

        return SHC(
            _values=self._values - other._values,
            normalization=self.normalization,
            dates=self.dates,
            attrs=self.attrs.copy(),
        )

    def de_mean(self, inplace: bool = False) -> "SHC":
        obj = self if inplace else self.copy()
        obj._values = obj._values - obj.mean
        return obj

    def replace(
            self,
            *args,
            inplace: bool = False,
    ) -> "SHC":
        """
        Replace one or multiple spherical harmonic coefficients.

        Parameters
        ----------
        *args :
            Replacement arguments.

            Supported usages:

            1. Replace one coefficient:

                shc.replace("c2,0", c20)

            2. Replace multiple coefficients:

                shc.replace(
                    "c1,0", c10,
                    "c1,1", c11,
                    "s1,1", s11,
                )

            3. Replace coefficients using dict:

                shc.replace({
                    "c1,0": c10,
                    "c1,1": c11,
                    "s1,1": s11,
                })

            Each replacement value can be SHC, ndarray, list, tuple or scalar.

        inplace : bool
            Whether to modify current object.

        Returns
        -------
        SHC
            SHC object with replaced coefficients.
        """

        from collections.abc import Mapping
        import numpy as np

        obj = self if inplace else self.copy()

        # --------------------------------------------------
        # 1. Parse replacement pairs
        # --------------------------------------------------
        if len(args) == 0:
            raise ValueError("No replacement arguments provided.")

        # Case 1: replace({"c1,0": c10, "c1,1": c11})
        if len(args) == 1 and isinstance(args[0], Mapping):
            pairs = list(args[0].items())

        # Case 2: replace([("c1,0", c10), ("c1,1", c11)])
        elif (
                len(args) == 1
                and isinstance(args[0], (list, tuple))
                and all(
            isinstance(item, (list, tuple)) and len(item) == 2
            for item in args[0]
        )
        ):
            pairs = list(args[0])

        # Case 3: replace("c1,0", c10, "c1,1", c11, ...)
        else:
            if len(args) % 2 != 0:
                raise ValueError(
                    "Replacement arguments should be given as pairs: "
                    "index1, value1, index2, value2, ..."
                )

            pairs = list(zip(args[0::2], args[1::2]))

        # --------------------------------------------------
        # 2. Apply replacements
        # --------------------------------------------------
        for index, new in pairs:
            if not isinstance(index, str):
                raise TypeError(
                    f"Coefficient index should be str, got {type(index)}."
                )

            try:
                index1d = MathTool.get_cs_1d_index(index)
            except Exception as e:
                warnings.warn(str(e))
                continue

            if index1d >= obj._values.shape[1]:
                raise IndexError(
                    f"Coefficient index {index} exceeds SHC coefficient size."
                )

            # new is another SHC
            if isinstance(new, SHC):
                if index1d >= new._values.shape[1]:
                    raise IndexError(
                        f"Coefficient index {index} exceeds replacement SHC size."
                    )
                new_array = np.asarray(new._values[:, index1d], dtype=float)

            # new is ndarray/list/scalar
            else:
                new_array = np.asarray(new, dtype=float)

            new_array = np.squeeze(new_array)

            # scalar replacement
            if new_array.ndim == 0:
                new_array = np.full(obj.ntime, float(new_array))

            # time series replacement
            elif new_array.ndim == 1:
                if new_array.size != obj.ntime:
                    raise ValueError(
                        f"Replacement length for {index} is {new_array.size}, "
                        f"but SHC time length is {obj.ntime}."
                    )

            else:
                raise ValueError(
                    f"Replacement value for {index} should be scalar or 1-D array, "
                    f"got shape {new_array.shape}."
                )

            # only replace finite values
            valid = np.isfinite(new_array)
            obj._values[valid, index1d] = new_array[valid]

        return obj

    # ------------------------------------------------------------------
    # Filtering wrapper
    # ------------------------------------------------------------------
    def _old_filter(
            self,
            *args,
            params: tuple | None = None,
            inplace: bool = False,
            **kwargs,
    ) -> "SHC":
        """
        Apply one or multiple SHC filters.

        Supported usages
        ----------------
        1. Old style, one filter:

            shc.filter(SHCDecorrelationType.PnMm, (3, 10))

        2. Old style with keyword params:

            shc.filter(SHCDecorrelationType.PnMm, params=(3, 10))

        3. Multiple filters, flat style:

            shc.filter(
                SHCDecorrelationType.PnMm, (3, 10),
                SHCFilterType.DDK, (3,),
            )

        4. Multiple filters, tuple style:

            shc.filter(
                (SHCDecorrelationType.PnMm, (3, 10)),
                (SHCFilterType.DDK, (3,)),
            )

        5. Multiple filters, list style:

            shc.filter([
                (SHCDecorrelationType.PnMm, (3, 10)),
                (SHCFilterType.DDK, (3,)),
            ])

        Parameters
        ----------
        *args :
            Filter specifications.

        params : tuple or None
            Keyword-style parameters for backward compatibility.

        inplace : bool
            Whether to modify current object.

        **kwargs :
            Extra keyword arguments passed to every filter.

        Returns
        -------
        SHC
            Filtered SHC object.
        """

        from collections.abc import Mapping
        from sagea.filtering.factory import apply_filter_to_cs

        def _is_filter_spec(x) -> bool:
            return isinstance(x, (list, tuple)) and 1 <= len(x) <= 3

        def _parse_one_spec(spec):
            """
            Parse:
                (method,)
                (method, params)
                (method, params, kwargs)
            """
            if not _is_filter_spec(spec):
                raise TypeError(
                    "Each filter specification should be "
                    "(method,), (method, params), or (method, params, kwargs)."
                )

            method = spec[0]
            filter_params = spec[1] if len(spec) >= 2 else None

            if len(spec) == 3:
                local_kwargs = spec[2]
                if not isinstance(local_kwargs, Mapping):
                    raise TypeError(
                        "The third item of a filter specification should be a dict-like object."
                    )
                local_kwargs = dict(local_kwargs)
            else:
                local_kwargs = {}

            return method, filter_params, local_kwargs

        def _parse_filter_args(args):
            if len(args) == 0:
                raise ValueError("No filter method provided.")

            # Case 1:
            # shc.filter(method, params=(...))
            if params is not None:
                if len(args) != 1:
                    raise ValueError(
                        "When using keyword argument params=..., only one method "
                        "should be provided."
                    )
                return [(args[0], params, {})]

            # Case 2:
            # shc.filter([ (method1, params1), (method2, params2) ])
            if (
                    len(args) == 1
                    and isinstance(args[0], (list, tuple))
                    and len(args[0]) > 0
                    and all(_is_filter_spec(item) for item in args[0])
            ):
                return [_parse_one_spec(item) for item in args[0]]

            # Case 3:
            # shc.filter((method1, params1), (method2, params2))
            if all(_is_filter_spec(item) for item in args):
                return [_parse_one_spec(item) for item in args]

            # Case 4:
            # shc.filter(method)
            if len(args) == 1:
                return [(args[0], None, {})]

            # Case 5:
            # shc.filter(method1, params1, method2, params2, ...)
            if len(args) % 2 != 0:
                raise ValueError(
                    "Filter arguments should be given as pairs: "
                    "method1, params1, method2, params2, ..."
                )

            specs = []
            for i in range(0, len(args), 2):
                method = args[i]
                filter_params = args[i + 1]
                specs.append((method, filter_params, {}))

            return specs

        obj = self if inplace else self.copy()

        filter_specs = _parse_filter_args(args)

        for method, filter_params, local_kwargs in filter_specs:
            current_kwargs = dict(kwargs)
            current_kwargs.update(local_kwargs)

            obj._values = apply_filter_to_cs(
                cs=obj._values,
                method=method,
                params=filter_params,
                lmax=obj.lmax,
                **current_kwargs,
            )

        return obj

    @cached_property
    def filter(self) -> _SHCFilterAccessor:
        """
        Filter accessor.
        Recommended usage
        -----------------
        shc.filter.some_method(...)
        Deprecated usage
        ----------------
        shc.filter("some_method", ...)
        """
        return _SHCFilterAccessor(self)

    # ------------------------------------------------------------------
    # Converting wrapper
    # ------------------------------------------------------------------
    def _normalize_physical_dimension(self, field_type):
        """
        Normalize physical dimension input.

        Accept:
        - PhysicalDimension.Geopotential
        - "Geopotential"
        - "EWH"
        """

        from sagea.constants.constant import PhysicalDimension

        if isinstance(field_type, PhysicalDimension):
            return field_type

        if isinstance(field_type, str):
            if field_type in PhysicalDimension.__members__:
                return PhysicalDimension[field_type]

            for item in PhysicalDimension:
                if str(item.value) == field_type:
                    return item

        raise ValueError(f"Invalid PhysicalDimension: {field_type}")

    def _load_love_number(
            self,
            lmax: int,
            love_number_method=None,
    ):
        """
        Load Love numbers for SHC physical conversion.
        """

        try:
            from sagea.physics.love_number import LoveNumber
        except ImportError as exc:
            raise ImportError(
                "Love number is required for SHC physical conversion. "
                "Please provide `ln=...`, or make sure `physics.love_number.LoveNumber` exists."
            ) from exc

        love = LoveNumber()
        love.configuration.set_lmax(lmax)

        if love_number_method is not None:
            love.configuration.set_method(love_number_method)

        return love.get_Love_number()

    def convert(
            self,
            from_type,
            to_type,
            ln=None,
            love_number_method=None,
            inplace: bool = False,
    ):
        """
        Convert SHC coefficients between physical dimensions.

        Parameters
        ----------
        from_type : PhysicalDimension or str
            Input physical dimension.

        to_type : PhysicalDimension or str
            Output physical dimension.

        ln : ndarray or None
            Load Love numbers.
            If None, this method will try to load Love numbers automatically.

        love_number_method : optional
            Love number method. Only used when ln is None.

        inplace : bool
            If True, modify current SHC.
            If False, return a new SHC object.

        Returns
        -------
        SHC
            Converted SHC object.

        Examples
        --------
        shc_ewh = shc.convert(
            from_type=PhysicalDimension.Geopotential,
            to_type=PhysicalDimension.EWH,
            ln=ln,
        )
        """

        import copy

        from sagea.physics.shc_convert import ConvertSHC, ConvertSHCConfig

        from_type = self._normalize_physical_dimension(from_type)
        to_type = self._normalize_physical_dimension(to_type)

        if from_type == to_type:
            if inplace:
                return self
            return copy.deepcopy(self)

        if ln is None:
            ln = self._load_love_number(
                lmax=self.lmax,
                love_number_method=love_number_method,
            )

        config = (
            ConvertSHCConfig()
            .set_input_type(from_type)
            .set_output_type(to_type)
            .set_Love_number(ln)
        )

        converter = ConvertSHC().config(config)

        converted_values = converter.apply_to(self._values)

        if inplace:
            self._values = converted_values
            return self

        out = copy.deepcopy(self)
        out._values = converted_values

        return out

    # ------------------------------------------------------------------
    # Geometric correction wrapper
    # ------------------------------------------------------------------
    def geometric_correct(
            self,
            assumption="ActualEarth",
            field_type="EWH",
            love_number=None,
            love_number_method="Wang",
            grid_space: float | None = 0.5,
            grid_type=None,
            orography=None,
            undulation=None,

            # 新增
            phisfc_file: str | None = None,
            gif48_file: str | None = None,
            geoid_lmax: int = 160,
            auto_load_actual_earth: bool = False,

            iter_max: int = 4,
            vmax: float = 2.5,
            vmin: float = 0.0,
            inplace: bool = False,
            log: bool = False,
    ):
        from sagea.corrections.geometric_correction.geometric import (
            apply_geometric_correction,
        )

        obj = self if inplace else self.copy()

        if love_number is None:
            love_number = obj._load_love_number(
                lmax=obj.lmax,
                love_number_method=love_number_method,
            )

        cqlm, sqlm = obj.cs2d

        cqlm_corr, sqlm_corr = apply_geometric_correction(
            cqlm=cqlm,
            sqlm=sqlm,
            love_number=love_number,
            assumption=assumption,
            field_type=field_type,
            love_number_method=love_number_method,
            grid_space=grid_space,
            grid_type=grid_type,
            orography=orography,
            undulation=undulation,

            phisfc_file=phisfc_file,
            gif48_file=gif48_file,
            geoid_lmax=geoid_lmax,
            auto_load_actual_earth=auto_load_actual_earth,

            iter_max=iter_max,
            vmax=vmax,
            vmin=vmin,
            log=log,
        )

        obj._values = MathTool.cs_combine_to_triangle_1d(cqlm_corr, sqlm_corr)

        return obj

    # ------------------------------------------------------------------
    # Harmonic synthesis wrapper
    # ------------------------------------------------------------------
    def to_grid(self, grid_space, grid_type=None):
        """
        SHC -> Grid.
        """
        from sagea.harmonics.transform import shc_to_grid
        from sagea.core.grid import GRD

        cqlm, sqlm = self.cs2d

        grid_value, lat, lon = shc_to_grid(
            cqlm=cqlm, sqlm=sqlm,
            lmax=self.lmax,
            grid_space=grid_space,
            grid_type=grid_type,
        )

        return GRD(grid_value, lat, lon)
