// Copyright (c) 2019 CNES
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.
#pragma once
#include <algorithm>
#include <boost/geometry.hpp>
#include <optional>
#include "pyinterp/detail/geometry/box.hpp"
#include "pyinterp/detail/geometry/point.hpp"

namespace pyinterp::detail::geometry {

/// Index points in the Cartesian space at N dimensions.
///
/// @tparam Coordinate The class of storage for a point's coordinates.
/// @tparam Type The type of data stored in the tree.
/// @tparam N Number of dimensions in the Cartesian space handled.
template <typename Coordinate, typename Type, size_t N>
class RTree {
 public:
  /// Type of distances between two points.
  using distance_t = typename boost::geometry::default_distance_result<
      geometry::PointND<Coordinate, N>, geometry::PointND<Coordinate, N>>::type;

  /// Type of query results.
  using result_t = std::pair<distance_t, Type>;

  /// Value handled by this object
  using value_t = std::pair<geometry::PointND<Coordinate, N>, Type>;

  /// Spatial index used
  using rtree_t =
      boost::geometry::index::rtree<value_t, boost::geometry::index::rstar<16>>;

  /// Default constructor
  RTree() : tree_(new rtree_t{}) {}

  /// Default destructor
  virtual ~RTree() = default;

  /// Default copy constructor
  RTree(const RTree &) = default;

  /// Default copy assignment operator
  RTree &operator=(const RTree &) = default;

  /// Move constructor
  RTree(RTree &&) noexcept = default;

  /// Move assignment operator
  RTree &operator=(RTree &&) noexcept = default;

  /// Returns the box able to contain all values stored in the container.
  ///
  /// @returns The box able to contain all values stored in the container or an
  /// invalid box if there are no values in the container.
  virtual std::optional<geometry::BoxND<Coordinate, N>> bounds() const {
    if (empty()) {
      return {};
    }
    return tree_->bounds();
  }

  /// Returns the number of points of this mesh
  ///
  /// @return the number of points
  [[nodiscard]] inline size_t size() const { return tree_->size(); }

  /// Query if the container is empty.
  ///
  /// @return true if the container is empty.
  [[nodiscard]] inline bool empty() const { return tree_->empty(); }

  /// Removes all values stored in the container.
  inline void clear() { tree_->clear(); }

  /// The tree is created using packing algorithm (The old data is erased before
  /// construction.)
  ///
  /// @param points
  void packing(const std::vector<value_t> &points) { *tree_ = rtree_t(points); }

  /// Insert new data into the search tree
  ///
  /// @param point
  void insert(const value_t &value) { tree_->insert(value); }

  /// Search for the K nearest neighbors of a given point.
  ///
  /// @param point Point of interest
  /// @param k The number of nearest neighbors to search.
  /// @return the k nearest neighbors
  std::vector<result_t> query(const geometry::PointND<Coordinate, N> &point,
                              const uint32_t k) const {
    auto result = std::vector<result_t>();
    std::for_each(
        tree_->qbegin(boost::geometry::index::nearest(point, k)), tree_->qend(),
        [&point, &result](const auto &item) {
          result.emplace_back(std::make_pair(
              boost::geometry::distance(point, item.first), item.second));
        });
    return result;
  }

  /// Search for the nearest neighbors of a given point within a radius r.
  ///
  /// @param point Point of interest
  /// @param radius distance within which neighbors are returned
  /// @return the k nearest neighbors
  template <
      int DimensionCount = N,
      typename std::enable_if<DimensionCount == 2, void>::type * = nullptr>
  std::vector<result_t> query_ball(
      const geometry::PointND<Coordinate, N> &point,
      const double radius) const {
    auto box = geometry::BoxND<Coordinate, N>(
        {point.template get<0>() - radius, point.template get<1>() - radius},
        {point.template get<0>() + radius, point.template get<1>() + radius});
    auto within = std::vector<value_t>();
    tree_->query(boost::geometry::index::covered_by(box),
                 std::back_inserter(within));
    auto result = std::vector<result_t>();
    for (auto &&item : within) {
      auto distance = boost::geometry::distance(point, item.first);
      if (distance <= radius) {
        result.emplace_back(std::make_pair(distance, item.second));
      }
    }
    return result;
  }

  template <
      int DimensionCount = N,
      typename std::enable_if<DimensionCount == 3, void>::type * = nullptr>
  std::vector<result_t> query_ball(
      const geometry::PointND<Coordinate, N> &point,
      const double radius) const {
    auto cube = RTree::cube(
        geometry::PointND<Coordinate, N>(point.template get<0>() - radius,
                                         point.template get<1>() - radius,
                                         point.template get<2>() - radius),
        geometry::PointND<Coordinate, N>(point.template get<0>() + radius,
                                         point.template get<1>() + radius,
                                         point.template get<2>() + radius));
    auto within = std::vector<value_t>();
    tree_->query(boost::geometry::index::covered_by(cube),
                 std::back_inserter(within));
    auto result = std::vector<result_t>();
    // for (auto &&item : within) {
    //   auto distance = boost::geometry::distance(point, item.first);
    //   if (distance <= radius) {
    //     result.emplace_back(std::make_pair(distance, item.second));
    //   }
    // }
    return result;
  }

  template <int DimensionCount = N,
            typename std::enable_if<DimensionCount != 2 && DimensionCount != 3,
                                    void>::type * = nullptr>
  std::vector<result_t> query_ball(
      const geometry::PointND<Coordinate, N> &point,
      const double radius) const {
    auto result = std::vector<result_t>();
    std::for_each(
        tree_->qbegin(boost::geometry::index::satisfies([&](const auto &item) {
          return boost::geometry::distance(item.first, point) <= radius;
        })),
        tree_->qend(), [&point, &result](const auto &item) {
          result.emplace_back(std::make_pair(
              boost::geometry::distance(point, item.first), item.second));
        });
    return result;
  }

  /// Search for the nearest K neighbors around a given point.
  ///
  /// @param point Point of interest
  /// @param k The number of nearest neighbors to search.
  /// @return the k nearest neighbors if the point is within by its
  /// neighbors.
  std::vector<result_t> query_within(
      const geometry::PointND<Coordinate, N> &point, const uint32_t k) const {
    auto result = std::vector<result_t>();
    auto points =
        boost::geometry::model::multi_point<geometry::PointND<Coordinate, N>>();
    points.reserve(k);

    std::for_each(
        tree_->qbegin(boost::geometry::index::nearest(point, k)), tree_->qend(),
        [&points, &point, &result](const auto &item) {
          points.emplace_back(item.first);
          result.emplace_back(std::make_pair(
              boost::geometry::distance(point, item.first), item.second));
        });

    // Are found points located around the requested point?
    if (!boost::geometry::covered_by(
            point,
            boost::geometry::return_envelope<
                boost::geometry::model::box<geometry::PointND<Coordinate, N>>>(
                points))) {
      return {};
    }
    return result;
  }

 protected:
  /// Geographic index used to store data and their searches.
  std::shared_ptr<rtree_t> tree_;

 private:
  /// Polygon used
  using Polygon =
      boost::geometry::model::polygon<geometry::PointND<Coordinate, N>>;

  /// Multi-polygon used
  using MultiPolygon = boost::geometry::model::multi_polygon<Polygon>;

  /// Defines from two points a collection of contiguous 3D polygons that share
  /// certain edges and describe a cube.
  static MultiPolygon cube(const geometry::PointND<Coordinate, N> &min_corner,
                           const geometry::PointND<Coordinate, N> &max_corner) {
    return MultiPolygon(
        {// bottom (0 0 0, 0 1 0, 1 1 0, 1 0 0, 0 0 0),
         Polygon{{{min_corner.template get<0>(), min_corner.template get<1>(),
                   min_corner.template get<2>()},
                  {min_corner.template get<0>(), max_corner.template get<1>(),
                   min_corner.template get<2>()},
                  {max_corner.template get<0>(), max_corner.template get<1>(),
                   min_corner.template get<2>()},
                  {max_corner.template get<0>(), min_corner.template get<1>(),
                   min_corner.template get<2>()},
                  {min_corner.template get<0>(), min_corner.template get<1>(),
                   min_corner.template get<2>()}}},
         // left (0 0 0, 0 0 1, 0 1 1, 0 1 0, 0 0 0),
         Polygon{{{min_corner.template get<0>(), min_corner.template get<1>(),
                   min_corner.template get<2>()},
                  {min_corner.template get<0>(), min_corner.template get<1>(),
                   max_corner.template get<2>()},
                  {min_corner.template get<0>(), max_corner.template get<1>(),
                   max_corner.template get<2>()},
                  {min_corner.template get<0>(), max_corner.template get<1>(),
                   min_corner.template get<2>()},
                  {min_corner.template get<0>(), min_corner.template get<1>(),
                   min_corner.template get<2>()}}},
         // front (0 0 0, 1 0 0, 1 0 1, 0 0 1, 0 0 0),
         Polygon{{{min_corner.template get<0>(), min_corner.template get<1>(),
                   min_corner.template get<2>()},
                  {max_corner.template get<0>(), min_corner.template get<1>(),
                   min_corner.template get<2>()},
                  {max_corner.template get<0>(), min_corner.template get<1>(),
                   max_corner.template get<2>()},
                  {min_corner.template get<0>(), min_corner.template get<1>(),
                   max_corner.template get<2>()},
                  {min_corner.template get<0>(), min_corner.template get<1>(),
                   min_corner.template get<2>()}}},
         // top (0 0 1, 1 0 1, 1 1 1, 0 1 1, 0 0 1),
         Polygon{{{min_corner.template get<0>(), min_corner.template get<1>(),
                   max_corner.template get<2>()},
                  {max_corner.template get<0>(), min_corner.template get<1>(),
                   max_corner.template get<2>()},
                  {max_corner.template get<0>(), max_corner.template get<1>(),
                   max_corner.template get<2>()},
                  {min_corner.template get<0>(), max_corner.template get<1>(),
                   max_corner.template get<2>()},
                  {min_corner.template get<0>(), min_corner.template get<1>(),
                   max_corner.template get<2>()}}},
         // right (1 0 0, 1 1 0, 1 1 1, 1 0 1, 1 0 0),
         Polygon{{{max_corner.template get<0>(), min_corner.template get<1>(),
                   min_corner.template get<2>()},
                  {max_corner.template get<0>(), max_corner.template get<1>(),
                   min_corner.template get<2>()},
                  {max_corner.template get<0>(), max_corner.template get<1>(),
                   max_corner.template get<2>()},
                  {max_corner.template get<0>(), min_corner.template get<1>(),
                   max_corner.template get<2>()},
                  {max_corner.template get<0>(), min_corner.template get<1>(),
                   min_corner.template get<2>()}}},
         // Back (1 1 0, 0 1 0, 0 1 1, 1 1 1, 1 1 0)
         Polygon{{{max_corner.template get<0>(), max_corner.template get<1>(),
                   min_corner.template get<2>()},
                  {min_corner.template get<0>(), max_corner.template get<1>(),
                   min_corner.template get<2>()},
                  {min_corner.template get<0>(), max_corner.template get<1>(),
                   max_corner.template get<2>()},
                  {max_corner.template get<0>(), max_corner.template get<1>(),
                   max_corner.template get<2>()},
                  {max_corner.template get<0>(), max_corner.template get<1>(),
                   min_corner.template get<2>()}}}});
  }
};

}  // namespace pyinterp::detail::geometry
