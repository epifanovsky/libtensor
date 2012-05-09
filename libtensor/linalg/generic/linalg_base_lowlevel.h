#ifndef LIBTENSOR_LINALG_BASE_LOWLEVEL_H
#define LIBTENSOR_LINALG_BASE_LOWLEVEL_H

namespace libtensor {


/** \brief Provides low-level (Levels 1 through 3) linear algebra routines
    \tparam M Memory driver.
    \tparam L1 Level-1 driver.
    \tparam L2 Level-2 driver.
    \tparam L3 Level-3 driver.

    \ingroup libtensor_linalg
 **/
template<typename M, typename L1, typename L2, typename L3>
struct linalg_base_lowlevel :
    public M,
    public L1,
    public L2,
    public L3
{ };


} // namespace libtensor

#endif // LIBTENSOR_LINALG_BASE_LOWLEVEL_H
