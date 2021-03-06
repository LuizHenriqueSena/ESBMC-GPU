#include <iostream>
#include <cassert>
#include <set>
using namespace std;

int main ()
{
  int myints[]={10,12,75};
  int myints2[]={20,25,32};
  
  multiset<int> first (myints,myints+3);     // 10,12,75
  multiset<int> second (myints2,myints2+3);  // 20,25,32
  multiset<int>::iterator it;

  assert(first.size() == 3);
  it = first.begin();
  assert(*it == 10);
  it++;
  assert(*it == 12);
  it++;
  assert(*it == 75);
  it++;
  assert(second.size() == 3);
  it = second.begin();
  assert(*it == 20);
  it++;
  assert(*it == 25);
  it++;
  assert(*it == 32);
  it++;

  first.swap(second);
  assert(first.size() == 3);
  it = first.begin();
  assert(*it != 20);
  it++;
  assert(*it != 25);
  it++;
  assert(*it == 32);
  it++;
  assert(second.size() != 3);
  it = second.begin();
  assert(*it == 10);
  it++;
  assert(*it == 12);
  it++;
  assert(*it == 75);
  it++;

  cout << "first contains:";
  for (it=first.begin(); it!=first.end(); it++) cout << " " << *it;

  cout << "\nsecond contains:";
  for (it=second.begin(); it!=second.end(); it++) cout << " " << *it;

  cout << endl;

  return 0;
}
