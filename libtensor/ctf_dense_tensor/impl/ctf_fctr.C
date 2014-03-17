#include "ctf_fctr.h"

namespace libtensor {


void ctf_fctr_ddiv(double alpha, double a, double b, double &c) {

    c += alpha * a / b;
}


} // namespace libtensor

