#include <stddef.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <tuple>

#include "SpMM.hh"

template<typename Mtx>
tuple<float *, int, int> extract_matrix_data(const Mtx &mtx) {
    auto [rows, cols] = mtx.size();
    auto data = (float *)malloc(sizeof(float) * rows * cols);
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            data[i * rows + j] = mtx.at(i, j);
        }
    }
    return std::make_tuple(data, rows, cols);
}

typedef struct {
    int column;
    float value;
} SparseMatrixElement;

typedef struct {
    int size;
    SparseMatrixElement *elements;
} SparseMatrixRow;

static SparseMatrixRow *compress_sparse_matrix(const SparseMatrix &mtx) {
    int non_zero_element_num = 0;

    auto [rows, cols] = mtx.size();

    SparseMatrixRow *result = (SparseMatrixRow *)malloc(sizeof(SparseMatrixRow) * rows);

    std::vector<std::tuple<int, int, int, float> > vec;
    vec.reserve(rows * cols / 32);

    for (int i = 0; i < rows; ++i) {
        result[i].size = 0;
        for (int j = 0; j < cols; ++j) {
            float data = mtx.at(i, j);
            if (data != 0.0) {
                vec.push_back(make_tuple(i, j, result[i].size, data));
                result[i].size += 1;
            }
        }
        non_zero_element_num += result[i].size;
    }

    SparseMatrixElement *elements = (SparseMatrixElement *)malloc(
            sizeof(SparseMatrixElement) * non_zero_element_num);

    for (int i = 0; i < rows; ++i) {
        result[i].elements = elements;
        elements += result[i].size;
    }

    for (auto coord : vec) {
        auto [i, j, cursor, data] = coord;
        result[i].elements[cursor].column = j;
        result[i].elements[cursor].value = data;
    }

    return result;
}

static void matrix_transpose(float *result, float *data, int rows, int cols) {
    for (int i = 0; i < rows; i += 8) {
        for (int j = 0; j < cols; j += 8) {
            for (int ii = i; ii < rows && ii < i + 8; ++ii) {
                for (int jj = j; jj < cols && jj < j + 8; ++jj) {
                    result[jj * rows + ii] = data[ii * rows + jj];
                }
            }
        }
    }
}

static void spmm_core(float *result, const Matrix &A, const SparseMatrix &B) {
    auto [mtx_a_, A_rows, A_cols] = extract_matrix_data(A);
    auto [B_rows, B_cols] = B.size();
    auto sparse_b = compress_sparse_matrix(B);

    float *mtx_a = (float *)malloc(sizeof(float) * A_rows * A_cols);
    matrix_transpose(mtx_a, mtx_a_, A_rows, A_cols);

    float *data = (float *)malloc(sizeof(float) * A_rows * B_cols);
    memset(data, 0, sizeof(float) * A_rows * B_rows);

    for (int j = 0; j < B_rows; ++j) {
        for (int k = 0; k < sparse_b[j].size; ++k) {
            auto s = sparse_b[j].elements[k];
            auto dst = data + j * A_rows;
            auto src = mtx_a + s.column * A_rows;

            int i;

            for (i = 0; i < A_rows && ((uintptr_t)(dst + i) & 0x3f); ++i) {
                dst[i] += src[i] * s.value;
            }

            for (; i < (A_rows / 16 - 1) * 16; i += 16) {
                dst[i + 0] += src[i + 0] * s.value;
                dst[i + 1] += src[i + 1] * s.value;
                dst[i + 2] += src[i + 2] * s.value;
                dst[i + 3] += src[i + 3] * s.value;
                dst[i + 4] += src[i + 4] * s.value;
                dst[i + 5] += src[i + 5] * s.value;
                dst[i + 6] += src[i + 6] * s.value;
                dst[i + 7] += src[i + 7] * s.value;
                dst[i + 8] += src[i + 8] * s.value;
                dst[i + 9] += src[i + 9] * s.value;
                dst[i + 10] += src[i + 10] * s.value;
                dst[i + 11] += src[i + 11] * s.value;
                dst[i + 12] += src[i + 12] * s.value;
                dst[i + 13] += src[i + 13] * s.value;
                dst[i + 14] += src[i + 14] * s.value;
                dst[i + 15] += src[i + 15] * s.value;
            }

            for (; i < A_rows; ++i) {
                dst[i] += src[i] * s.value;
            }
        }
    }

    matrix_transpose(result, data, A_rows, B_rows);

    free(mtx_a_);
    free(mtx_a);
    free(data);
    free(sparse_b[0].elements);
    free(sparse_b);
}

Matrix SpMM_opt(const Matrix &A, const SparseMatrix &B) {
    if (A.size() != B.size()) {
        return Matrix();
    }

    auto [A_rows, A_cols] = A.size();
    auto [B_rows, B_cols] = B.size();

    std::vector<float> result(A_rows * B_rows);
    spmm_core(result.data(), A, B);
    return Matrix(result, A_rows, B_rows);
}
