#include <cmath>
#include "../test_utils.h"

bool cmp(double diff, double ref) {

    const double k_thresh = 1e-12;

    if(fabs(ref) > 1.0) return fabs(diff) < fabs(ref) * k_thresh;
    else return fabs(diff) < k_thresh;
}

