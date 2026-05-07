#pragma once

#include "shared.h"

#include <algorithm>
#include <cstddef>
#include <initializer_list>
#include <utility>
#include <vector>

namespace intmasseval {

class poly_t {
public:
	std::vector<float> poly;

	poly_t() = default;
	explicit poly_t(std::size_t size) : poly(size, 0.0f) {}
	explicit poly_t(const std::vector<float> &values) : poly(values) {}
	poly_t(std::initializer_list<float> values) : poly(values) {}

	std::size_t size() const {
		return poly.size();
	}

	float &operator[](std::size_t idx) {
		return poly[idx];
	}

	const float &operator[](std::size_t idx) const {
		return poly[idx];
	}
};

class offset_poly_t {
public:
	int offset;
	poly_t poly;

	offset_poly_t() : offset(0), poly() {}
	offset_poly_t(int offset_in, const poly_t &poly_in) : offset(offset_in), poly(poly_in) {}
};

using peak_t = std::pair<int, float>;
using peaklist_t = std::vector<peak_t>;
using frag_poly_t = std::vector<std::pair<formula_t, offset_poly_t>>;

inline bool sort_peak_desc(const peak_t &a, const peak_t &b);

struct spectrum_t {
	formula_t formula;
	peaklist_t peaks;

	void sort_peaks() {
		std::sort(peaks.begin(), peaks.end(), sort_peak_desc);
	}
};

inline bool sort_peak_desc(const peak_t &a, const peak_t &b) {
	if (a.second == b.second) {
		return a.first < b.first;
	}
	return a.second > b.second;
}

offset_poly_t poly_mul(const offset_poly_t &a, const offset_poly_t &b);
peaklist_t get_mass_peaks(const formula_t &formula);
peaklist_t poly_to_peaks(const offset_poly_t &p);
frag_poly_t get_all_frag_poly(const formula_t &formula);
std::vector<spectrum_t> get_all_frag_spect(const formula_t &formula);
offset_poly_t poly_coalesce(const offset_poly_t &a, float threshold);

}  // namespace intmasseval
