void f(unsigned int counter) {
  if(counter==0) return;

  f(counter-1);
}

int main() {
  unsigned int x;
  __ESBMC_assume(x<=10);
  
  f(x);

}
