#!/usr/bin/env python3
import time, random, math, sys
from vdb import VectorDatabase, VDBMetric

try:
  import matplotlib.pyplot as plt
  import matplotlib.gridspec as gridspec
except ImportError:
  import subprocess
  subprocess.check_call([sys.executable, "-m", "pip", "install", "matplotlib", "--quiet"])
  import matplotlib.pyplot as plt
  import matplotlib.gridspec as gridspec

def rv(d): return [random.gauss(0,1) for _ in range(d)]

def benchmark_insertion(dims, counts):
  print(f"\n{'='*60}\nBENCHMARK: Insertion Performance\n{'='*60}")
  r={'counts':[],'times':[],'throughput':[]}
  for c in counts:
    print(f"  Testing {c:,} vectors with {dims} dimensions...")
    db=VectorDatabase(dims,VDBMetric.COSINE)
    v=[rv(dims) for _ in range(c)]
    s=time.time()
    for i,x in enumerate(v): db.add_vector(x,f"vec_{i}")
    e=time.time()-s; t=c/e
    r['counts']+= [c]; r['times']+= [e]; r['throughput']+= [t]
    print(f"    Time: {e:.3f}s, Throughput: {t:.0f} vectors/sec")
    del db
  return r

def benchmark_dimensionality(ds,fc=1000):
  print(f"\n{'='*60}\nBENCHMARK: Dimensionality Impact\n{'='*60}")
  r={'dimensions':[],'insert_time':[],'search_time':[]}
  for d in ds:
    print(f"  Testing {d} dimensions with {fc} vectors...")
    db=VectorDatabase(d,VDBMetric.COSINE)
    v=[rv(d) for _ in range(fc)]
    s=time.time()
    for i,x in enumerate(v): db.add_vector(x,f"vec_{i}")
    it=time.time()-s
    q=rv(d); s=time.time()
    for _ in range(50): db.search(q,10)
    st=(time.time()-s)/50*1000
    r['dimensions']+=[d]; r['insert_time']+=[it]; r['search_time']+=[st]
    print(f"    Insert: {it:.3f}s, Search: {st:.2f}ms")
    del db
  return r

def benchmark_scalability(bd=64,ms=[1,2,4,8,16]):
  print(f"\n{'='*60}\nBENCHMARK: Scalability Test (Memory & Performance)\n{'='*60}")
  r={'total_vectors':[],'memory_mb':[],'insert_throughput':[],'search_latency':[]}
  for m in ms:
    d=bd*m; c=max(100,10000//m)
    print(f"  Testing {c:,} vectors × {d} dims = {c*d:,} total elements")
    db=VectorDatabase(d,VDBMetric.COSINE)
    v=[rv(d) for _ in range(c)]
    s=time.time()
    for i,x in enumerate(v): db.add_vector(x,f"vec_{i}")
    it=time.time()-s; tp=c/it
    q=rv(d); s=time.time()
    for _ in range(20): db.search(q,5)
    sl=(time.time()-s)/20*1000
    me=(c*d*4+c*100)/(1024*1024)
    r['total_vectors']+=[c*d]; r['memory_mb']+=[me]; r['insert_throughput']+=[tp]; r['search_latency']+=[sl]
    print(f"    Throughput: {tp:.0f} vec/s, Latency: {sl:.2f}ms")
    del db
  return r

def create_comprehensive_plots(ar):
  print(f"\n{'='*60}\nGenerating visualization...\n{'='*60}")
  fig=plt.figure(figsize=(18,6))
  gs=gridspec.GridSpec(1,3,figure=fig,hspace=0.3,wspace=0.3)

  ax1=fig.add_subplot(gs[0,0]); ins=ar['insertion']
  ax1.plot(ins['counts'],ins['times'],'o-',linewidth=3,markersize=10,color='#2E86AB',label='Time')
  ax1.set(xlabel='Number of Vectors',ylabel='Time (seconds)',title='Insertion Performance')
  ax1.grid(True,alpha=0.3,linestyle='--'); ax1.set_xscale('log')
  ax1t=ax1.twinx()
  ax1t.plot(ins['counts'],ins['throughput'],'s--',linewidth=2,markersize=8,color='#A23B72',label='Throughput',alpha=0.7)
  ax1t.set_ylabel('Throughput (vectors/sec)',color='#A23B72'); ax1t.tick_params(axis='y',labelcolor='#A23B72')
  ax1.legend(loc='upper left'); ax1t.legend(loc='upper right')

  ax2=fig.add_subplot(gs[0,1]); d=ar['dimensionality']
  ax2.plot(d['dimensions'],d['insert_time'],'^-',linewidth=3,markersize=10,color='#F18F01',label='Insert')
  ax2t=ax2.twinx()
  ax2t.plot(d['dimensions'],d['search_time'],'v-',linewidth=3,markersize=10,color='#C73E1D',label='Search')
  ax2.set(xlabel='Dimensions',ylabel='Insert Time (s)',title='Impact of Dimensionality')
  ax2.grid(True,alpha=0.3,linestyle='--'); ax2.tick_params(axis='y',labelcolor='#F18F01')
  ax2t.set_ylabel('Search Time (ms)',color='#C73E1D'); ax2t.tick_params(axis='y',labelcolor='#C73E1D')
  ax2.legend(loc='upper left'); ax2t.legend(loc='upper right')

  ax3=fig.add_subplot(gs[0,2]); s=ar['scalability']
  sc=ax3.scatter(s['total_vectors'],s['memory_mb'],s=300,c=s['search_latency'],cmap='viridis',alpha=0.8,edgecolors='black',linewidth=2)
  for i,x in enumerate(s['total_vectors']):
    ax3.annotate(f"{s['memory_mb'][i]:.1f}MB\n{s['insert_throughput'][i]:.0f}v/s",(x,s['memory_mb'][i]),textcoords="offset points",xytext=(0,12),ha='center',fontsize=9)
  ax3.set(xlabel='Total Vector Elements',ylabel='Memory Usage (MB)',title='Scalability: Memory vs Size')
  ax3.set_xscale('log'); ax3.grid(True,alpha=0.3,linestyle='--')
  plt.colorbar(sc,ax=ax3).set_label('Search Latency (ms)')

  plt.suptitle('vdb vector database - performance benchmarks',fontsize=16,y=1.02)
  of='benchmark.png'; plt.savefig(of,dpi=150,bbox_inches='tight',facecolor='white')
  print(f"  Saved plot to: {of}")
  return of

def main():
  print("\n"+"="*60+"\n  vdb stress tests\n"+"="*60)
  ar={}
  ar['insertion']=benchmark_insertion(128,[100,500,1000,5000,10000,25000])
  ar['dimensionality']=benchmark_dimensionality([32,64,128,256,512,1024],1000)
  ar['scalability']=benchmark_scalability(64,[1,2,4,8,16])
  pf=create_comprehensive_plots(ar)
  print(f"\n{'='*60}\n  completed...\n{'='*60}\n")
  return pf

if __name__=="__main__": main()
