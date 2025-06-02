# uHDR: HDR image editing software
#   Copyright (C) 2021  remi cozot 
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
# hdrCore project 2020
# author: remi.cozot@univ-littoral.fr

# -----------------------------------------------------------------------------
# --- Package hdrCore ---------------------------------------------------------
# -----------------------------------------------------------------------------
"""
package hdrCore consists of the core classes for HDR imaging.
"""

# -----------------------------------------------------------------------------
# --- Import ------------------------------------------------------------------
# -----------------------------------------------------------------------------
import numba
import numpy as np
# -----------------------------------------------------------------------------
# --- Functions: numba version ------------------------------------------------
# -----------------------------------------------------------------------------
@numba.jit(cache=True, parallel=True)
def numba_cctf_sRGB_encoding(L):
    """cctf SRGB encoding (numba acceleration)

        Args:
            L (float or numpy.ndarray, Required)

        Returns:
            (float)
    """
    v = np.where(L <= 0.0031308, L * 12.92, 1.055 * np.power(L, 1 / 2.4) - 0.055)
    return v
# -----------------------------------------------------------------------------
@numba.jit(cache=True, parallel=True)
def numba_cctf_sRGB_decoding(V):
    """cctf SRGB decoding (numba acceleration)

        Args:
            V (float or numpy.ndarray, Required)

        Returns:
            (float)
    """
    L = np.where(V <= numba_cctf_sRGB_encoding(0.0031308), V / 12.92, np.power((V + 0.055) / 1.055, 2.4))
    return L

# -----------------------------------------------------------------------------
# --- Functions: cuda version ------------------------------------------------
# -----------------------------------------------------------------------------
@numba.vectorize('float32(float32)', target='cuda' )
def cuda_cctf_sRGB_decoding(V):
    """cctf sRGB decoding (cuda acceleration)

        Args:
            V (float or numpy.ndarray, Required)

        Returns:
            (float)
    """

    if V <= 0.040449935999999999:
        L = V / 12.92
    else:
        L = ((V + 0.055) / 1.055)**( 2.4)
    return L
# -----------------------------------------------------------------------------
@numba.vectorize('float32(float32)', target='cuda' )
def cuda_cctf_sRGB_encoding(L):
    """cctf SRGB encoding (cuda acceleration)

        Args:
            L (float or numpy.ndarray, Required)

        Returns:
            (float)
    """

    if L <= 0.0031308:
        v = L * 12.92
    else:
        v = 1.055 * (L**(1 / 2.4)) - 0.055
    return v
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
def numba_sRGB_to_XYZ(sRGB, cctf_decoding=None):
    pass
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
#CAT_CAT02 = np.array([
#    [0.7328, 0.4296, -0.1624],
#    [-0.7036, 1.6975, 0.0061],
#    [0.0030, 0.0136, 0.9834],
#])
## -----------------------------------------------------------------------------
#def RGB_to_XYZ(RGB, illuminant_RGB, illuminant_XYZ, RGB_to_XYZ_matrix, chromatic_adaptation_transform='CAT02', cctf_decoding=None,**kwargs):
 
#    cctf_decoding = handle_arguments_deprecation({
#        'ArgumentRenamed': [['decoding_cctf', 'cctf_decoding']],
#    }, **kwargs).get('cctf_decoding', cctf_decoding)

#    RGB = to_domain_1(RGB)

#    if cctf_decoding is not None:
#        with domain_range_scale('ignore'):
#            RGB = cctf_decoding(RGB)

#    XYZ = dot_vector(RGB_to_XYZ_matrix, RGB)

#    if chromatic_adaptation_transform is not None:
#        M_CAT = chromatic_adaptation_matrix_VonKries(
#            xyY_to_XYZ(xy_to_xyY(illuminant_RGB)),
#            xyY_to_XYZ(xy_to_xyY(illuminant_XYZ)),
#            transform=chromatic_adaptation_transform)

#        XYZ = dot_vector(M_CAT, XYZ)

#    return from_range_1(XYZ)
## -----------------------------------------------------------------------------
#def RGB_to_RGB_matrix(input_colourspace,output_colourspace, chromatic_adaptation_transform='CAT02'):
#    M = input_colourspace.RGB_to_XYZ_matrix

#    if chromatic_adaptation_transform is not None:
#        M_CAT = chromatic_adaptation_matrix_VonKries(
#            xy_to_XYZ(input_colourspace.whitepoint),
#            xy_to_XYZ(output_colourspace.whitepoint),
#            chromatic_adaptation_transform)

#        M = dot_matrix(M_CAT, input_colourspace.RGB_to_XYZ_matrix)

#    M = dot_matrix(output_colourspace.XYZ_to_RGB_matrix, M)

#    return M
## -----------------------------------------------------------------------------
#def intermediate_luminance_function_CIE1976(f_Y_Y_n, Y_n=100):

#    f_Y_Y_n = as_float_array(f_Y_Y_n)
#    Y_n = as_float_array(Y_n)

#    Y = as_float(
#        np.where(
#            f_Y_Y_n > 24 / 116,
#            Y_n * f_Y_Y_n ** 3,
#            Y_n * (f_Y_Y_n - 16 / 116) * (108 / 841),
#        ))

#    return Y
## -----------------------------------------------------------------------------
#def Lab_to_XYZ(Lab, illuminant=CCS_ILLUMINANTS['CIE 1931 2 Degree Standard Observer']['D65']):

#    L, a, b = tsplit(to_domain_100(Lab))

#    X_n, Y_n, Z_n = tsplit(xyY_to_XYZ(xy_to_xyY(illuminant)))

#    f_Y_Y_n = (L + 16) / 116
#    f_X_X_n = a / 500 + f_Y_Y_n
#    f_Z_Z_n = f_Y_Y_n - b / 200

#    X = intermediate_luminance_function_CIE1976(f_X_X_n, X_n)
#    Y = intermediate_luminance_function_CIE1976(f_Y_Y_n, Y_n)
#    Z = intermediate_luminance_function_CIE1976(f_Z_Z_n, Z_n)

#    XYZ = tstack([X, Y, Z])

#    return from_range_1(XYZ)
## -----------------------------------------------------------------------------
#def XYZ_to_Lab(XYZ, illuminant=CCS_ILLUMINANTS['CIE 1931 2 Degree Standard Observer']['D65']):
 
#    X, Y, Z = tsplit(to_domain_1(XYZ))

#    X_n, Y_n, Z_n = tsplit(xyY_to_XYZ(xy_to_xyY(illuminant)))

#    f_X_X_n = intermediate_lightness_function_CIE1976(X, X_n)
#    f_Y_Y_n = intermediate_lightness_function_CIE1976(Y, Y_n)
#    f_Z_Z_n = intermediate_lightness_function_CIE1976(Z, Z_n)

#    L = 116 * f_Y_Y_n - 16
#    a = 500 * (f_X_X_n - f_Y_Y_n)
#    b = 200 * (f_Y_Y_n - f_Z_Z_n)

#    Lab = tstack([L, a, b])

#    return from_range_100(Lab)
## -----------------------------------------------------------------------------
