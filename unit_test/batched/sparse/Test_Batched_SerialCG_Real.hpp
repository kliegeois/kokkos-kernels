
#if defined(KOKKOSKERNELS_INST_FLOAT)
TEST_F( TestCategory, batched_scalar_serial_CG_float ) {
  test_batched_CG<TestExecSpace,float>();
}
#endif

#if defined(KOKKOSKERNELS_INST_DOUBLE)
TEST_F( TestCategory, batched_scalar_serial_CG_double ) {
  test_batched_CG<TestExecSpace,double>();
}
#endif
