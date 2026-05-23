import socket,threading,time
def pipe(a,b):
    try:
        while True:
            d=a.recv(4096)
            if not d:break
            b.sendall(d)
    except:pass
    finally:
        [x.close() for x in [a,b] if x]
def proxy(lp,rp):
    s=socket.socket();s.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1)
    s.bind(('0.0.0.0',lp));s.listen(20)
    while True:
        c,_=s.accept();u=socket.socket();u.connect(('127.0.0.1',rp))
        [threading.Thread(target=pipe,args=x,daemon=True).start() for x in [(c,u),(u,c)]]
for lp,rp in [(15432,25432),(16379,26379)]:
    threading.Thread(target=proxy,args=(lp,rp),daemon=True).start()
while True:time.sleep(60)
