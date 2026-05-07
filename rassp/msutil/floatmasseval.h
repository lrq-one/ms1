#pragma once

#include "shared.h"
#include "intmasseval.h"

#include <algorithm>
#include <cstddef>
#include <utility>
#include <vector>

namespace floatmasseval {

struct fspectrum_t {
	formula_t formula;
	std::vector<std::pair<float, float>> peaks;

	void sort_peaks() {
		std::sort(peaks.begin(), peaks.end(), [](const auto &a, const auto &b) {
			if (a.second == b.second) {
				return a.first < b.first;
			}
			return a.second > b.second;
		});
	}
};

namespace detail {
inline void generate_sub_formulae_impl(
	const formula_t &formula,
	std::size_t idx,
	formula_t &current,
	std::vector<formula_t> &out
) {
	if (idx >= NUM_FORMULA_ELEMENTS) {
		out.push_back(current);
		return;
	}

	for (std::size_t count = 0; count <= formula.counts[idx]; ++count) {
		current.counts[idx] = count;
		generate_sub_formulae_impl(formula, idx + 1, current, out);
	}

	current.counts[idx] = 0;
}
}  // namespace detail

inline std::vector<formula_t> generate_sub_formulae(const formula_t &formula) {
	std::vector<formula_t> out;
	formula_t current;
	detail::generate_sub_formulae_impl(formula, 0, current, out);
	return out;
}

inline std::vector<fspectrum_t> get_all_frag_fspect(const formula_t &formula) {
	std::vector<fspectrum_t> out;
	const std::vector<intmasseval::spectrum_t> int_spect = intmasseval::get_all_frag_spect(formula);
	out.reserve(int_spect.size());
	for (const auto &item : int_spect) {
		fspectrum_t converted;
		converted.formula = item.formula;
		converted.peaks.reserve(item.peaks.size());
		for (const auto &peak : item.peaks) {
			converted.peaks.emplace_back(static_cast<float>(peak.first), peak.second);
		}
		converted.sort_peaks();
		out.push_back(std::move(converted));
	}
	return out;
}

}  // namespace floatmasseval
