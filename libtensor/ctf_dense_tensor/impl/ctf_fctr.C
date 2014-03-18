#include "ctf_fctr.h"

namespace libtensor {


void ctf_fctr_ddiv(double alpha, double a, double b, double &c) {

    c += alpha * a / b;
}


void ctf_fsum_dmul(double alpha, double a, double &b) {

    b = alpha * a * b;
}


void ctf_fsum_dmul_add(double alpha, double a, double &b) {

    b += alpha * a * b;
}


void ctf_fsum_ddiv(double alpha, double a, double &b) {

    b = alpha * b / a;
}


void ctf_fsum_ddiv_add(double alpha, double a, double &b) {

    b += alpha * b / a;
}


} // namespace libtensor

