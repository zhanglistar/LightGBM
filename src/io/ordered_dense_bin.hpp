#ifndef LIGHTGBM_IO_ORDERED_DENSE_BIN_HPP_
#define LIGHTGBM_IO_ORDERED_DENSE_BIN_HPP_

#include <LightGBM/bin.h>

#include <cstring>
#include <cstdint>

#include <vector>
#include <mutex>
#include <algorithm>

#include "dense_bin.hpp"

namespace LightGBM {

template <typename VAL_T>
class OrderedDenseBin : public OrderedBin {
public:

  OrderedDenseBin(const DenseBin<VAL_T>* bin_data)
    :bin_data_(bin_data) {
    ordered_bin_.resize(bin_data_->num_data_);
  }

  ~OrderedDenseBin() {
  }

  void Init(const char* is_index_used, int num_leaves) override {
    // initialize the leaf information
    leaf_start_ = std::vector<data_size_t>(num_leaves, 0);
    leaf_cnt_ = std::vector<data_size_t>(num_leaves, 0);
    if (is_index_used == nullptr) {
      std::memcpy(ordered_bin_.data(), bin_data_->data_.data(), sizeof(VAL_T)*bin_data_->num_data_);
      leaf_cnt_[0] = static_cast<data_size_t>(bin_data_->num_data_);
    } else {
      // if using part of data(bagging)
      data_size_t j = 0;
      for (int i = 0; i < bin_data_->num_data_; ++i) {
        if (is_index_used[i]) {
          ordered_bin_[j] = bin_data_->data_[i];
          ++j;
        }
      }
      leaf_cnt_[0] = j;
    }
  }

  void ConstructHistogram(int leaf, const score_t* ordered_gradients, const score_t* ordered_hessians,
                          HistogramBinEntry* out) const override {
    // get current leaf boundary
    const int rest = leaf_cnt_[leaf] & 0x3;
    data_size_t i = 0;
    const data_size_t start = leaf_start_[leaf];
    // use data on current leaf to construct histogram
    for (; i < leaf_cnt_[leaf] - rest; i += 4) {

      const VAL_T bin0 = ordered_bin_[start + i];
      const VAL_T bin1 = ordered_bin_[start + i + 1];
      const VAL_T bin2 = ordered_bin_[start + i + 2];
      const VAL_T bin3 = ordered_bin_[start + i + 3];

      out[bin0].sum_gradients += ordered_gradients[i];
      out[bin1].sum_gradients += ordered_gradients[i + 1];
      out[bin2].sum_gradients += ordered_gradients[i + 2];
      out[bin3].sum_gradients += ordered_gradients[i + 3];

      out[bin0].sum_hessians += ordered_hessians[i];
      out[bin1].sum_hessians += ordered_hessians[i + 1];
      out[bin2].sum_hessians += ordered_hessians[i + 2];
      out[bin3].sum_hessians += ordered_hessians[i + 3];

      ++out[bin0].cnt;
      ++out[bin1].cnt;
      ++out[bin2].cnt;
      ++out[bin3].cnt;
    }

    for (; i < leaf_cnt_[leaf]; ++i) {
      const VAL_T bin0 = ordered_bin_[start + i];
      out[bin0].sum_gradients += ordered_gradients[i];
      out[bin0].sum_hessians += ordered_hessians[i];
      ++out[bin0].cnt;
    }
  }

  void ConstructHistogram(int leaf, const score_t* ordered_gradients,
                          HistogramBinEntry* out) const override {
    // get current leaf boundary
    const int rest = leaf_cnt_[leaf] & 0x3;
    data_size_t i = 0;
    const data_size_t start = leaf_start_[leaf];
    // use data on current leaf to construct histogram
    for (; i < leaf_cnt_[leaf] - rest; i += 4) {

      const VAL_T bin0 = ordered_bin_[start + i];
      const VAL_T bin1 = ordered_bin_[start + i + 1];
      const VAL_T bin2 = ordered_bin_[start + i + 2];
      const VAL_T bin3 = ordered_bin_[start + i + 3];

      out[bin0].sum_gradients += ordered_gradients[i];
      out[bin1].sum_gradients += ordered_gradients[i + 1];
      out[bin2].sum_gradients += ordered_gradients[i + 2];
      out[bin3].sum_gradients += ordered_gradients[i + 3];

      ++out[bin0].cnt;
      ++out[bin1].cnt;
      ++out[bin2].cnt;
      ++out[bin3].cnt;
    }

    for (; i < leaf_cnt_[leaf]; ++i) {
      const VAL_T bin0 = ordered_bin_[start + i];
      out[bin0].sum_gradients += ordered_gradients[i];
      ++out[bin0].cnt;
    }
  }

  void Split(int leaf, int right_leaf, const data_size_t* indices, data_size_t new_left_cnt) override {
    // get current leaf boundary
    const data_size_t l_start = leaf_start_[leaf];
    const data_size_t l_end = l_start + leaf_cnt_[leaf];
    const data_size_t r_cnt = leaf_cnt_[leaf] - new_left_cnt;
    std::vector<VAL_T> tmp(r_cnt);
    // new left leaf end after split
    data_size_t new_left_end = l_start;
    data_size_t new_right_cnt = 0;
    for (data_size_t i = l_start; i < l_end; ++i) {
      if (indices[i - l_start] >=0) {
        ordered_bin_[new_left_end++] = ordered_bin_[i];
      } else {
        tmp[new_right_cnt++] = ordered_bin_[i];
      }
    }
    for (data_size_t i = 0; i < new_right_cnt; ++i) {
      ordered_bin_[new_left_end + i] = tmp[i];
    }
    leaf_start_[right_leaf] = new_left_end;
    leaf_cnt_[leaf] = new_left_end - l_start;
    CHECK(leaf_cnt_[leaf] == new_left_cnt);
    leaf_cnt_[right_leaf] = new_right_cnt;
  }

  data_size_t NonZeroCount(int leaf) const override {
    return static_cast<data_size_t>(leaf_cnt_[leaf]);
  }
  /*! \brief Disable copy */
  OrderedDenseBin<VAL_T>& operator=(const OrderedDenseBin<VAL_T>&) = delete;
  /*! \brief Disable copy */
  OrderedDenseBin<VAL_T>(const OrderedDenseBin<VAL_T>&) = delete;

private:
  const DenseBin<VAL_T>* bin_data_;
  std::vector<VAL_T> ordered_bin_;
  /*! \brief leaf_start_[i] means data in i-th leaf start from */
  std::vector<data_size_t> leaf_start_;
  /*! \brief leaf_cnt_[i] means number of data in i-th leaf */
  std::vector<data_size_t> leaf_cnt_;
};

template <typename VAL_T>
OrderedBin* DenseBin<VAL_T>::CreateOrderedBin() const {
  return new OrderedDenseBin<VAL_T>(this);
}

}  // namespace LightGBM
#endif   // LIGHTGBM_IO_ORDERED_DENSE_BIN_HPP_
