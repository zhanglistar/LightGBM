#ifndef LIGHTGBM_IO_ORDERED_SPARSE_BIN_HPP_
#define LIGHTGBM_IO_ORDERED_SPARSE_BIN_HPP_

#include <LightGBM/bin.h>

#include <cstring>
#include <cstdint>

#include <vector>
#include <mutex>
#include <algorithm>

#include "sparse_bin.hpp"

namespace LightGBM {

/*!
* \brief Interface for ordered bin data. efficient for construct histogram, especially for sparse bin
*        There are 2 advantages by using ordered bin.
*        1. group the data by leafs to improve the cache hit.
*        2. only store the non-zero bin, which can speed up the histogram consturction for sparse features.
*        However it brings additional cost: it need re-order the bins after every split, which will cost much for dense feature.
*        So we only using ordered bin for sparse situations.
*/
template <typename VAL_T>
class OrderedSparseBin: public OrderedBin {
public:

  OrderedSparseBin(const SparseBin<VAL_T>* bin_data)
    :bin_data_(bin_data) {
    data_size_t cur_pos = 0;
    data_size_t i_delta = -1;
    int non_zero_cnt = 0;
    while (bin_data_->NextNonzero(&i_delta, &cur_pos)) {
      ++non_zero_cnt;
    }
    ordered_bin_.resize(non_zero_cnt);
    ordered_indices_.resize(non_zero_cnt);
    leaf_cnt_.push_back(non_zero_cnt);
  }

  ~OrderedSparseBin() {
  }

  void Init(const char* is_index_used, int num_leaves) override {
    // initialize the leaf information
    leaf_start_.resize(num_leaves, 0);
    leaf_cnt_.resize(num_leaves, 0);
    if (is_index_used == nullptr) {
      // if using all data, copy all non-zero pair
      data_size_t j = 0;
      data_size_t cur_pos = 0;
      data_size_t i_delta = -1;
      while (bin_data_->NextNonzero(&i_delta, &cur_pos)) {
        ordered_indices_[j] = cur_pos;
        ordered_bin_[j] = bin_data_->vals_[i_delta];
        ++j;
      }
      leaf_cnt_[0] = static_cast<data_size_t>(j);
    } else {
      // if using part of data(bagging)
      data_size_t j = 0;
      data_size_t cur_pos = 0;
      data_size_t i_delta = -1;
      while (bin_data_->NextNonzero(&i_delta, &cur_pos)) {
        if (is_index_used[cur_pos]) {
          ordered_indices_[j] = cur_pos;
          ordered_bin_[j] = bin_data_->vals_[i_delta];
          ++j;
        }
      }
      leaf_cnt_[0] = j;
    }
  }

  void ConstructHistogram(int leaf, const score_t* ordered_gradients, const score_t* ordered_hessians,
                          HistogramBinEntry* out) const override {
    // get current leaf boundary
    const data_size_t start = leaf_start_[leaf];
    const data_size_t end = start + leaf_cnt_[leaf];
    const int rest = leaf_cnt_[leaf] & 0x3;
    data_size_t i = start;
    // use data on current leaf to construct histogram
    for (; i < end - rest; i += 4) {

      const VAL_T bin0 = ordered_bin_[i];
      const VAL_T bin1 = ordered_bin_[i + 1];
      const VAL_T bin2 = ordered_bin_[i + 2];
      const VAL_T bin3 = ordered_bin_[i + 3];

      out[bin0].sum_gradients += ordered_gradients[ordered_indices_[i]];
      out[bin1].sum_gradients += ordered_gradients[ordered_indices_[i + 1]];
      out[bin2].sum_gradients += ordered_gradients[ordered_indices_[i + 2]];
      out[bin3].sum_gradients += ordered_gradients[ordered_indices_[i + 3]];

      out[bin0].sum_hessians += ordered_hessians[ordered_indices_[i]];
      out[bin1].sum_hessians += ordered_hessians[ordered_indices_[i + 1]];
      out[bin2].sum_hessians += ordered_hessians[ordered_indices_[i + 2]];
      out[bin3].sum_hessians += ordered_hessians[ordered_indices_[i + 3]];

      ++out[bin0].cnt;
      ++out[bin1].cnt;
      ++out[bin2].cnt;
      ++out[bin3].cnt;
    }

    for (; i < end; ++i) {
      const VAL_T bin0 = ordered_bin_[i];
      out[bin0].sum_gradients += ordered_gradients[ordered_indices_[i]];
      out[bin0].sum_hessians += ordered_hessians[ordered_indices_[i]];
      ++out[bin0].cnt;
    }

  }

  void ConstructHistogram(int leaf, const score_t* ordered_gradients,
                          HistogramBinEntry* out) const override {
    // get current leaf boundary
    const data_size_t start = leaf_start_[leaf];
    const data_size_t end = start + leaf_cnt_[leaf];
    const int rest = leaf_cnt_[leaf] & 0x3;
    data_size_t i = start;
    // use data on current leaf to construct histogram
    for (; i < end - rest; i += 4) {

      const VAL_T bin0 = ordered_bin_[i];
      const VAL_T bin1 = ordered_bin_[i + 1];
      const VAL_T bin2 = ordered_bin_[i + 2];
      const VAL_T bin3 = ordered_bin_[i + 3];

      out[bin0].sum_gradients += ordered_gradients[ordered_indices_[i]];
      out[bin1].sum_gradients += ordered_gradients[ordered_indices_[i + 1]];
      out[bin2].sum_gradients += ordered_gradients[ordered_indices_[i + 2]];
      out[bin3].sum_gradients += ordered_gradients[ordered_indices_[i + 3]];


      ++out[bin0].cnt;
      ++out[bin1].cnt;
      ++out[bin2].cnt;
      ++out[bin3].cnt;
    }

    for (; i < end; ++i) {
      const VAL_T bin0 = ordered_bin_[i];
      out[bin0].sum_gradients += ordered_gradients[ordered_indices_[i]];
      ++out[bin0].cnt;
    }
  }

  void Split(int leaf, int right_leaf, const data_size_t* indices, data_size_t) override {
    // get current leaf boundary
    const data_size_t l_start = leaf_start_[leaf];
    const data_size_t l_end = l_start + leaf_cnt_[leaf];
    // new left leaf end after split
    data_size_t left = l_start;
    data_size_t right = l_end - 1;
    // To-do: try to partition to two sorted array
    while (left <= right) {
      while (left <= right && indices[ordered_indices_[left]] >= 0) {
        ordered_indices_[left] = indices[ordered_indices_[left]];
        ++left;
      }
      while (left <= right && indices[ordered_indices_[right]] < 0) {
        ordered_indices_[right] = ~indices[ordered_indices_[right]];
        --right;
      }
      if (left < right) {
        std::swap(ordered_bin_[left], ordered_bin_[right]);
        std::swap(ordered_indices_[left], ordered_indices_[right]);
        ordered_indices_[left] = indices[ordered_indices_[left]];
        ordered_indices_[right] = ~indices[ordered_indices_[right]];
        ++left;
        --right;
      }
    }
    leaf_start_[right_leaf] = left;
    leaf_cnt_[leaf] = left - l_start;
    leaf_cnt_[right_leaf] = l_end - left;
  }

  data_size_t NonZeroCount(int leaf) const override {
    return static_cast<data_size_t>(leaf_cnt_[leaf]);
  }
  /*! \brief Disable copy */
  OrderedSparseBin<VAL_T>& operator=(const OrderedSparseBin<VAL_T>&) = delete;
  /*! \brief Disable copy */
  OrderedSparseBin<VAL_T>(const OrderedSparseBin<VAL_T>&) = delete;

private:
  const SparseBin<VAL_T>* bin_data_;
  /*! \brief Store non-zero pair , group by leaf */
  std::vector<VAL_T> ordered_bin_;
  std::vector<data_size_t> ordered_indices_;
  /*! \brief leaf_start_[i] means data in i-th leaf start from */
  std::vector<data_size_t> leaf_start_;
  /*! \brief leaf_cnt_[i] means number of data in i-th leaf */
  std::vector<data_size_t> leaf_cnt_;
};

template <typename VAL_T>
OrderedBin* SparseBin<VAL_T>::CreateOrderedBin() const {
  return new OrderedSparseBin<VAL_T>(this);
}

}  // namespace LightGBM
#endif   // LightGBM_IO_ORDERED_SPARSE_BIN_HPP_
