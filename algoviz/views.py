from networkx.drawing.nx_pydot import graphviz_layout
from collections import deque
from django.shortcuts import render
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from PIL import Image
import os,io
from .models import pics
from django.core.files.images import ImageFile
import midea,random
import networkx as nx

def home(request):
    return render(request,'algoviz\home.html')

def binary(request):
    p=pics.objects.filter(name="binary_search")
    p.delete()
    error=""
    len1=request.GET.get('length')
    list1=request.GET.get('list')
    target=request.GET.get('tofind')
    if not(len1) and not(list1) and not(target):
        binary_search()
    else:
        if not(len1) or not(list1) or not(target):
            error="!!! ENTER ALL FIELDS !!!"
            binary_search()
        else:
            try:
                len1=int(len1)
                list1=list1.split(",")
                list1=list(map(int, list1))
                target=int(target)
                if(list1==sorted(list1) and len(list1)==len1 and target in list1):
                    binary_search(n=len1,y=list1,target=target)
                else:
                    binary_search()
                    error="!!! INVALID INPUT !!!"
            except ValueError:
                error="!!! INVALID INPUT !!!"
                binary_search()
    pic=pics.objects.filter(name="binary_search")
    f1=[]
    f2=[]
    for i in pic:
        if i.fno==1:
            f1.append(i)
        else:
            f2.append(i)
    return render(request,'algoviz\\binary.html',{'pic':pic,'f1':f1,'f2':f2,'error':error})

def linear(request):
    p=pics.objects.filter(name="linear_search")
    p.delete()
    error=""
    len1=request.GET.get('length')
    list1=request.GET.get('list')
    target=request.GET.get('tofind')
    if not(len1) and not(list1) and not(target):
        linear_search()
    else:
        if not(len1) or not(list1) or not(target):
            error="!!! ENTER ALL FIELDS !!!"
            linear_search()
        else:
            try:
                len1=int(len1)
                list1=list1.split(",")
                list1=list(map(int, list1))
                target=int(target)
                if(len(list1)==len1 and target in list1):
                    linear_search(n=len1,y=list1,target=target)
                else:
                    linear_search()
                    error="!!! INVALID INPUT !!!"
            except ValueError:
                error="!!! INVALID INPUT !!!"
                linear_search()
    pic=pics.objects.filter(name="linear_search")
    f1=[]
    f2=[]
    for i in pic:
        if i.fno==1:
            f1.append(i)
        else:
            f2.append(i)
    return render(request,'algoviz\linear.html',{'pic':pic,'f1':f1,'f2':f2,'error':error})

def dij(request):
    p=pics.objects.filter(name="dijkstra")
    p.delete()
    error=""
    l1=request.GET.get("length")
    l2=0
    list1=request.GET.get("list")
    if(not l1 and not list1):
        dijkstra()
    else:
        if not(l1) or not(list1):
            error="!!! ENTER ALL FIELDS !!!"
            dijkstra()
        else:
            try:
                l1=int(l1)
            except ValueError:
                error="Invalid Input"
            list1=list1[1:-1].split(");(")
            if(l1!=len(list1)):
                error="!!! INVALID INPUT !!!"
                dijkstra()
            else:
                try:
                    for i in range(l1):
                        list1[i]=list1[i].split(",")
                        list1[i]=list(map(int, list1[i]))
                    dijkstra(l1,list1)
                except ValueError:
                    error="Enter Fields in the format (int,int,int);(int,int,int);..."
                    plt.cla()
                    dijkstra()
                except nx.NetworkXError:
                    error="The node 0 is not in the graph."
                    plt.cla()
                    dijkstra()

    pic=pics.objects.filter(name="dijkstra")
    f1=[]
    f2=[]
    for i in pic:
        if i.fno==1:
            f1.append(i)
        else:
            f2.append(i)
    return render(request,'algoviz\dij.html',{'pic':pic,'f1':f1,'f2':f2,'error':error})

def df(request):
    p=pics.objects.filter(name="dfs")
    p.delete()
    l1=request.GET.get("length")
    l2=0
    error=""
    list1=request.GET.get("list")
    if(not l1 and not list1):
        dfs()
    else:
        if not(l1) or not(list1):
            error="!!! ENTER ALL FIELDS !!!"
            dfs()
        else:
            try:
                l1=int(l1)
            except ValueError:
                error="Invalid Input"
            list1=list1[1:-1].split(");(")
            if(l1!=len(list1)):
                error="!!! INVALID INPUT !!!"
                dfs()
            else:
                try:
                    for i in range(l1):
                        list1[i]=list1[i].split(",")
                        list1[i]=list(map(int, list1[i]))
                    dfs(l1,list1)
                except ValueError:
                    error="Enter Fields in the format (int,int);(int,int);..."
                    dfs()
    pic=pics.objects.filter(name="dfs")
    f1=[]
    f2=[]
    for i in pic:
        if i.fno==1:
            f1.append(i)
        else:
            f2.append(i)

    return render(request,'algoviz\df.html',{'pic':pic,'f1':f1,'f2':f2,'error':error})

def bf(request):
    p=pics.objects.filter(name="bfs")
    p.delete()
    l1=request.GET.get("length")
    l2=0
    error=""
    list1=request.GET.get("list")
    if(not l1 and not list1):
        bfs()
    else:
        if not(l1) or not(list1):
            error="!!! ENTER ALL FIELDS !!!"
            bfs()
        else:
            try:
                l1=int(l1)
            except ValueError:
                error="Invalid Input"
            list1=list1[1:-1].split(");(")
            if(l1!=len(list1)):
                error="!!! INVALID INPUT !!!"
                bfs()
            else:
                try:
                    for i in range(l1):
                        list1[i]=list1[i].split(",")
                        list1[i]=list(map(int, list1[i]))
                    bfs(l1,list1)
                except ValueError:
                    error="Enter Fields in the format (int,int);(int,int);..."
                    bfs()
    pic=pics.objects.filter(name="bfs")
    f1=[]
    f2=[]
    for i in pic:
        if i.fno==1:
            f1.append(i)
        else:
            f2.append(i)

    return render(request,'algoviz\\bf.html',{'pic':pic,'f1':f1,'f2':f2,'error':error})

def inod(request):
    p=pics.objects.filter(name="inorder")
    p.delete()
    l1=request.GET.get("length")
    error=""
    list1=request.GET.get("list")
    if(not l1 and not list1):
        inorder()
    else:
        if not(l1) or not(list1):
            error="!!! ENTER ALL FIELDS !!!"
            inorder()
        else:
            try:
                l1=int(l1)
            except ValueError:
                error="Invalid Input"
            list1=list1[1:-1].split(");(")
            if(l1!=len(list1)):
                error="!!! INVALID INPUT !!!"
                inorder()
            else:
                try:
                    for i in range(l1):
                        list1[i]=list1[i].split(",")
                        list1[i]=list(map(int, list1[i]))
                    inorder(l1,list1)
                except ValueError:
                    error="Enter Fields in the format (int,int);(int,int);..."
                    inorder()
    pic=pics.objects.filter(name="inorder")
    f1=[]
    f2=[]
    for i in pic:
        if i.fno==1:
            f1.append(i)
        else:
            f2.append(i)

    return render(request,'algoviz\\inod.html',{'pic':pic,'f1':f1,'f2':f2,'error':error})

def postod(request):
    p=pics.objects.filter(name="postorder")
    p.delete()
    l1=request.GET.get("length")
    error=""
    list1=request.GET.get("list")
    if(not l1 and not list1):
        postorder()
    else:
        if not(l1) or not(list1):
            error="!!! ENTER ALL FIELDS !!!"
            postorder()
        else:
            try:
                l1=int(l1)
            except ValueError:
                error="Invalid Input"
            list1=list1[1:-1].split(");(")
            if(l1!=len(list1)):
                error="!!! INVALID INPUT !!!"
                postorder()
            else:
                try:
                    for i in range(l1):
                        list1[i]=list1[i].split(",")
                        list1[i]=list(map(int, list1[i]))
                    postorder(l1,list1)
                except ValueError:
                    error="Enter Fields in the format (int,int);(int,int);..."
                    postorder()
    pic=pics.objects.filter(name="postorder")
    f1=[]
    f2=[]
    for i in pic:
        if i.fno==1:
            f1.append(i)
        else:
            f2.append(i)
    return render(request,'algoviz\postod.html',{'pic':pic,'f1':f1,'f2':f2,'error':error})

def preod(request):
    p=pics.objects.filter(name="preorder")
    p.delete()
    l1=request.GET.get("length")
    error=""
    list1=request.GET.get("list")
    if(not l1 and not list1):
        preorder()
    else:
        if not(l1) or not(list1):
            error="!!! ENTER ALL FIELDS !!!"
            preorder()
        else:
            try:
                l1=int(l1)
            except ValueError:
                error="Invalid Input"
            list1=list1[1:-1].split(");(")
            if(l1!=len(list1)):
                error="!!! INVALID INPUT !!!"
                preorder()
            else:
                try:
                    for i in range(l1):
                        list1[i]=list1[i].split(",")
                        list1[i]=list(map(int, list1[i]))
                    preorder(l1,list1)
                except ValueError:
                    error="Enter Fields in the format (int,int);(int,int);..."
                    preorder()
    pic=pics.objects.filter(name="preorder")
    f1=[]
    f2=[]
    for i in pic:
        if i.fno==1:
            f1.append(i)
        else:
            f2.append(i)
    return render(request,'algoviz\preod.html',{'pic':pic})

def binary_search(n=15,y = [12, 15, 21, 24, 42, 43, 45, 54, 61, 64, 65, 77, 81, 99, 100],target = random.choice([12, 15, 21, 24, 42, 43, 45, 54, 61, 64, 65, 77, 81, 99, 100])):
    output_gif = "output.gif"

    fig, ax = plt.subplots()
    x = np.linspace(1,n,n)

    l = 0
    r = len(y)-1

    i = 0
    brk = False
    while l<=r:
        figure3 = io.BytesIO()
        m = int(l+(r-l)/2)
        b1 = ax.bar(x[:m],y[:m],color='b')
        b2 = ax.bar(x[m],y[m],color='g')
        b3 = ax.bar(x[m+1:],y[m+1:],color='b')
        if y[m] == target:
            brk = True
        elif y[m] < target:
            l = m+1
        elif y[m] > target:
            r = m-1

        plt.title(f"step = {i+1}, m = {m}, element={y[m]}")
        plt.savefig(figure3,format="PNG")
        content_file = ImageFile(figure3)
        file_name=f"bframe{i}"
        plot_instance = pics(name="binary_search",fno=i+1)
        plot_instance.frame.save(file_name, content_file)
        plot_instance.save()
        i+=1
        if brk:
            break
    file=pics.objects.filter(name="binary_search")
    files = []
    for a in file:
        files.append("." + a.frame.url)
    figure = io.BytesIO()
    images = [Image.open(file) for file in files]
    images[0].save(figure, save_all=True, append_images=images[1:], duration=2000, loop=0,format="PNG")
    content_file = ImageFile(figure)
    plot_instance1 = pics(name="binary_search",fno=i+2)
    plot_instance1.gif.save(output_gif, content_file)
    plot_instance1.save()

def linear_search(n=15,y = [12, 15, 21, 24, 42, 43, 45, 54, 55, 56, 65, 77, 81, 99, 100],target = random.choice([12, 15, 21, 24, 42, 43, 45, 54, 61, 64, 65, 77, 81, 99, 100])):
    output_gif = "linearoutput.gif"
    fig, ax = plt.subplots()
    x = np.linspace(0,n,n)
    i = 0
    brk = False
    while i<len(y):
        figure2 = io.BytesIO()
        b1 = ax.bar(x[:i],y[:i],color='b')
        b2 = ax.bar(x[i],y[i],color='g')
        b3 = ax.bar(x[i+1:],y[i+1:],color='b')
        plt.title(f"step = {i+1}, element={y[i]}")
        plt.savefig(figure2,format="PNG")
        content_file = ImageFile(figure2)
        file_name=f"lframe{i}"
        plot_instance = pics(name="linear_search",fno=i+1)
        plot_instance.frame.save(file_name, content_file)
        plot_instance.save()
        if(y[i] == target):
            break
        i+=1

    file=pics.objects.filter(name="linear_search")
    files = []
    for a in file:
        files.append("." + a.frame.url)
    figure = io.BytesIO()
    images = [Image.open(file) for file in files]
    images[0].save(figure, save_all=True, append_images=images[1:], duration=2000, loop=0,format="PNG")
    content_file = ImageFile(figure)
    plot_instance1 = pics(name="linear_search",fno=i+2)
    plot_instance1.gif.save(output_gif, content_file)
    plot_instance1.save()

def dijkstra(l1=14,list1=[[0,1,4],[0,7,8],[1,7,11],[1,2,8],[2,8,2],[2,5,4],[2,3,7],[3,5,14],[3,4,9],[4,5,10],[5,6,2],[6,7,1],[6,8,6],[7,8,7]]):
    plt.cla()
    plt.switch_backend('agg')
    output_gif = "dijkstraoutput.gif"

    #Graph creation
    G = nx.Graph()
    for i in range(l1):
        G.add_edge(list1[i][0],list1[i][1],weight=list1[i][2])
    pos = nx.spring_layout(G, seed=7)

    ax = plt.gca()
    ax.margins(0.08)


    src = 0
    visited = []
    current = src
    distances = {node:None for node in G.nodes()}
    distances[src] = 0
    i=0
    while len(G.nodes()):
        figure1 = io.BytesIO()
        nx.draw_networkx_nodes(G,pos,node_size=700, node_color=['blue' if current == node else 'green' for node in G.nodes()])
        nx.draw_networkx_edges(G, pos,edgelist=G.edges(), width=3)
        nx.draw_networkx_labels(G, pos, labels={node:f"{node}\n{'dis: '+str(distances[node]) if distances[node] is not None else ''}" for node in G.nodes()}, font_size=14, font_family="sans-serif")
        nx.draw_networkx_edge_labels(G, pos, nx.get_edge_attributes(G, "weight"))
        plt.title(f"step = {i+1}")
        plt.savefig(figure1,format="PNG")
        content_file = ImageFile(figure1)
        file_name=f"dframe{i}"
        plot_instance2 = pics(name="dijkstra",fno=i+1)
        plot_instance2.frame.save(file_name, content_file)
        plot_instance2.save()

        visited.append(current)
        neighbors = G.neighbors(current)
        for n in neighbors:
            if n not in visited:
                m = distances[current] + G.get_edge_data(current,n)['weight']
                if distances[n] is None:
                    distances[n] = m
                else:
                    distances[n] = min(m,distances[n])
        try:
            current = min([dis for dis in distances if dis not in visited and distances[dis] is not None],key=distances.get)
        except ValueError:
            break
        i+=1
        plt.cla()
        plt.clf()
    file=pics.objects.filter(name="dijkstra")
    files = []
    for a in file:
        files.append("." + a.frame.url)
    figure2 = io.BytesIO()
    images = [Image.open(file) for file in files]
    images[0].save(figure2, save_all=True, append_images=images[1:], duration=2000, loop=0,format="PNG")
    content_file = ImageFile(figure2)
    plot_instance1 = pics(name="dijkstra",fno=i+2)
    plot_instance1.gif.save(output_gif, content_file)
    plot_instance1.save()

def dfs(l1=10,list1=[[1,2],[1,3],[2,4],[3,5],[4,5],[1,6],[5,6],[6,7],[7,8],[6,9]]):

    plt.cla()
    output_gif = "dfsoutput.gif"

    G = nx.Graph()
    for i in range(l1):
        G.add_edge(list1[i][0],list1[i][1])


    pos = nx.spring_layout(G, seed=7)

    ax = plt.gca()
    ax.margins(0.08)

    nodes = list(G.nodes())
    src_node = nodes[0]

    stack = deque()
    visited= []
    stack.append(src_node)

    i=0

    while(not len(stack)==0):
        figure1 = io.BytesIO()
        visiting = stack.pop()
        nbors = G.neighbors(visiting)

        for n in nbors:
            if n not in visited:
                stack.append(n)

        if(len(visited) == len(nodes)):
            break
        visited.append(visiting)
        nx.draw(G,pos,width=3,labels={node:node for node in nodes},node_color=['red' if node in visited else 'blue' for node in nodes])
        plt.title(f"step = {i+1}")
        plt.savefig(figure1,format="PNG")
        content_file = ImageFile(figure1)
        file_name=f"dfframe{i}"
        plot_instance2 = pics(name="dfs",fno=i+1)
        plot_instance2.frame.save(file_name, content_file)
        plot_instance2.save()
        plt.cla()
        i+=1

    file=pics.objects.filter(name="dfs")
    files = []
    for a in file:
        files.append("." + a.frame.url)
    figure2 = io.BytesIO()
    images = [Image.open(file) for file in files]
    images[0].save(figure2, save_all=True, append_images=images[1:], duration=2000, loop=0,format="PNG")
    content_file = ImageFile(figure2)
    plot_instance1 = pics(name="dfs",fno=i+2)
    plot_instance1.gif.save(output_gif, content_file)
    plot_instance1.save()

def bfs(l1=7,list1=[[1,2],[1,3],[2,4],[3,5],[4,5],[1,6],[5,6]]):

    figure1 = io.BytesIO()
    plt.cla()
    plt.switch_backend('agg')
    output_gif = "dfsoutput.gif"


    G = nx.Graph()
    for i in range(l1):
        G.add_edge(list1[i][0],list1[i][1])

    pos = nx.spring_layout(G, seed=7)

    ax = plt.gca()
    ax.margins(0.08)

    nodes = list(G.nodes())
    src_node = nodes[0]

    queue = deque()
    visited = [src_node]
    queue.append(src_node)

    i=0
    nx.draw(G,pos,width=3,labels={node:node for node in nodes},node_color=['red' if node in visited else 'blue' for node in nodes])
    plt.title(f"step = {i+1}")
    plt.savefig(figure1,format="PNG")
    content_file = ImageFile(figure1)
    file_name=f"dframe{i}"
    plot_instance2 = pics(name="bfs",fno=i+1)
    plot_instance2.frame.save(file_name, content_file)
    plot_instance2.save()
    i+=1
    while(not len(queue) == 0):

        visiting = queue.popleft()
        nbors = G.neighbors(visiting)

        if(len(visited) == len(nodes)):
            break

        for n in nbors:
            if n not in visited:
                figure1 = io.BytesIO()
                visited.append(n)
                queue.append(n)
                nx.draw(G,pos,width=3,labels={node:node for node in nodes},node_color=['red' if node in visited else 'blue' for node in nodes])
                plt.title(f"step = {i+1}")
                plt.savefig(figure1,format="PNG")
                content_file = ImageFile(figure1)
                file_name=f"dframe{i}"
                plot_instance2 = pics(name="bfs",fno=i+1)
                plot_instance2.frame.save(file_name, content_file)
                plot_instance2.save()
                plt.cla()
                i+=1

    #bfs graph traversal algorithm
    file=pics.objects.filter(name="bfs")
    files = []
    for a in file:
        files.append("." + a.frame.url)
    figure2 = io.BytesIO()
    images = [Image.open(file) for file in files]
    images[0].save(figure2, save_all=True, append_images=images[1:], duration=2000, loop=0,format="PNG")
    content_file = ImageFile(figure2)
    plot_instance1 = pics(name="bfs",fno=i+2)
    plot_instance1.gif.save(output_gif, content_file)
    plot_instance1.save()

def inorder(l1=5,list1=[[1,2],[1,3],[2,4],[2,5],[3,6]]):
    plt.cla()
    plt.switch_backend('agg')
    output_gif = "inorderoutput.gif"
    G = nx.DiGraph()

    root = 1
    for i in range(l1):
        G.add_edge(list1[i][0],list1[i][1])

    def hierarchy_pos(G, root=None, width=1., vert_gap = 0.2, vert_loc = 0, xcenter = 0.5):
        if not nx.is_tree(G):
            raise TypeError('cannot use hierarchy_pos on a graph that is not a tree')

        if root is None:
            if isinstance(G, nx.DiGraph):
                root = next(iter(nx.topological_sort(G)))
            else:
                root = random.choice(list(G.nodes))

        def _hierarchy_pos(G, root, width=1., vert_gap = 0.2, vert_loc = 0, xcenter = 0.5, pos = None, parent = None):

            if pos is None:
                pos = {root:(xcenter,vert_loc)}
            else:
                pos[root] = (xcenter, vert_loc)
            children = list(G.neighbors(root))
            if not isinstance(G, nx.DiGraph) and parent is not None:
                children.remove(parent)
            if len(children)!=0:
                dx = width/len(children)
                nextx = xcenter - width/2 - dx/2
                for child in children:
                    nextx += dx
                    pos = _hierarchy_pos(G,child, width = dx, vert_gap = vert_gap,
                                        vert_loc = vert_loc-vert_gap, xcenter=nextx,
                                        pos=pos, parent = root)
            return pos


        return _hierarchy_pos(G, root, width, vert_gap, vert_loc, xcenter)

    pos = hierarchy_pos(G,root)


    #code to validate and check if the given directional graph is in fact a binary tree
    def validate(G,root):
        x = list(G.neighbors(root))
        lx = len(x)
        return len(x)<=2 and all(validate(G,n) for n in x)



    def draw(i,cur):
        plt.cla()
        figure1 = io.BytesIO()
        nx.draw(G,pos,labels={node:node for node in G.nodes()},node_color=['red' if node==cur else 'blue' for node in G.nodes()])
        plt.title(f"step = {i+1}")
        plt.savefig(figure1,format="PNG")
        content_file = ImageFile(figure1)
        file_name=f"dframe{i}"
        plot_instance2 = pics(name="inorder",fno=i+1)
        plot_instance2.frame.save(file_name, content_file)
        plot_instance2.save()
        plt.cla()

    def inordertraversal(G,root,i=0):
        x = list(G.neighbors(root))
        lx = len(x)
        if(lx==0):
            draw(i,root)
            return i+1
        i = inordertraversal(G,x[0],i)
        plt.cla()
        plt.switch_backend('agg')
        draw(i,root)
        i+=1
        if(lx==2):
            i = inordertraversal(G,x[1],i)
        return i

    ino=inordertraversal(G,root)

    file=pics.objects.filter(name="inorder")
    files = []
    for a in file:
        files.append("." + a.frame.url)
    figure2 = io.BytesIO()
    images = [Image.open(file) for file in files]
    images[0].save(figure2, save_all=True, append_images=images[1:], duration=2000, loop=0,format="PNG")
    content_file = ImageFile(figure2)
    plot_instance1 = pics(name="inorder",fno=ino+2)
    plot_instance1.gif.save(output_gif, content_file)
    plot_instance1.save()

def postorder(l1=5,list1=[[1,2],[1,3],[2,4],[2,5],[3,6]]):
    plt.cla()
    plt.switch_backend('agg')
    output_gif = "postorderoutput.gif"
    G = nx.DiGraph()

    root = 1
    for i in range(l1):
        G.add_edge(list1[i][0],list1[i][1])

    def hierarchy_pos(G, root=None, width=1., vert_gap = 0.2, vert_loc = 0, xcenter = 0.5):
        if not nx.is_tree(G):
            raise TypeError('cannot use hierarchy_pos on a graph that is not a tree')

        if root is None:
            if isinstance(G, nx.DiGraph):
                root = next(iter(nx.topological_sort(G)))  #allows back compatibility with nx version 1.11
            else:
                root = random.choice(list(G.nodes))

        def _hierarchy_pos(G, root, width=1., vert_gap = 0.2, vert_loc = 0, xcenter = 0.5, pos = None, parent = None):

            if pos is None:
                pos = {root:(xcenter,vert_loc)}
            else:
                pos[root] = (xcenter, vert_loc)
            children = list(G.neighbors(root))
            if not isinstance(G, nx.DiGraph) and parent is not None:
                children.remove(parent)
            if len(children)!=0:
                dx = width/len(children)
                nextx = xcenter - width/2 - dx/2
                for child in children:
                    nextx += dx
                    pos = _hierarchy_pos(G,child, width = dx, vert_gap = vert_gap,
                                        vert_loc = vert_loc-vert_gap, xcenter=nextx,
                                        pos=pos, parent = root)
            return pos


        return _hierarchy_pos(G, root, width, vert_gap, vert_loc, xcenter)

    pos = hierarchy_pos(G,root)


    #code to validate and check if the given directional graph is in fact a binary tree
    def validate(G,root):
        x = list(G.neighbors(root))
        lx = len(x)
        return len(x)<=2 and all(validate(G,n) for n in x)


    def draw(i,cur):
        plt.cla()
        figure1 = io.BytesIO()
        nx.draw(G,pos,labels={node:node for node in G.nodes()},node_color=['red' if node==cur else 'blue' for node in G.nodes()])
        plt.title(f"step = {i+1}")
        plt.savefig(figure1,format="PNG")
        content_file = ImageFile(figure1)
        file_name=f"dframe{i}"
        plot_instance2 = pics(name="postorder",fno=i+1)
        plot_instance2.frame.save(file_name, content_file)
        plot_instance2.save()
        plt.cla()

    def postordertraversal(G,root,i=0):
        x=list(G.neighbors(root))
        lx=len(x)
        if lx>=1:
            i = postordertraversal(G,x[0],i)
        if lx==2:
            i = postordertraversal(G,x[1],i)
        draw(i,root)
        return i+1

    ino=postordertraversal(G,root)

    file=pics.objects.filter(name="postorder")
    files = []
    for a in file:
        files.append("." + a.frame.url)
    figure2 = io.BytesIO()
    images = [Image.open(file) for file in files]
    images[0].save(figure2, save_all=True, append_images=images[1:], duration=2000, loop=0,format="PNG")
    content_file = ImageFile(figure2)
    plot_instance1 = pics(name="postorder",fno=ino+2)
    plot_instance1.gif.save(output_gif, content_file)
    plot_instance1.save()

def preorder(l1=5,list1=[[1,2],[1,3],[2,4],[2,5],[3,6]]):
    plt.cla()
    plt.switch_backend('agg')
    output_gif = "preorderoutput.gif"
    G = nx.DiGraph()

    root = 1
    for i in range(l1):
        G.add_edge(list1[i][0],list1[i][1])

    def hierarchy_pos(G, root=None, width=1., vert_gap = 0.2, vert_loc = 0, xcenter = 0.5):
        if not nx.is_tree(G):
            raise TypeError('cannot use hierarchy_pos on a graph that is not a tree')

        if root is None:
            if isinstance(G, nx.DiGraph):
                root = next(iter(nx.topological_sort(G)))  #allows back compatibility with nx version 1.11
            else:
                root = random.choice(list(G.nodes))

        def _hierarchy_pos(G, root, width=1., vert_gap = 0.2, vert_loc = 0, xcenter = 0.5, pos = None, parent = None):

            if pos is None:
                pos = {root:(xcenter,vert_loc)}
            else:
                pos[root] = (xcenter, vert_loc)
            children = list(G.neighbors(root))
            if not isinstance(G, nx.DiGraph) and parent is not None:
                children.remove(parent)
            if len(children)!=0:
                dx = width/len(children)
                nextx = xcenter - width/2 - dx/2
                for child in children:
                    nextx += dx
                    pos = _hierarchy_pos(G,child, width = dx, vert_gap = vert_gap,
                                        vert_loc = vert_loc-vert_gap, xcenter=nextx,
                                        pos=pos, parent = root)
            return pos


        return _hierarchy_pos(G, root, width, vert_gap, vert_loc, xcenter)

    pos = hierarchy_pos(G,root)

    #code to validate and check if the given directional graph is in fact a binary tree
    def validate(G,root):
        x = list(G.neighbors(root))
        lx = len(x)
        return len(x)<=2 and all(validate(G,n) for n in x)

    def draw(i,cur):
        figure1 = io.BytesIO()
        nx.draw(G,pos,labels={node:node for node in G.nodes()},node_color=['red' if node==cur else 'blue' for node in G.nodes()])
        plt.title(f"step = {i+1}")
        plt.savefig(figure1,format="PNG")
        content_file = ImageFile(figure1)
        file_name=f"dframe{i}"
        plot_instance2 = pics(name="preorder",fno=i+1)
        plot_instance2.frame.save(file_name, content_file)
        plot_instance2.save()
        plt.cla()

    def preordertraversal(G,root,i=0):
        x=list(G.neighbors(root))
        lx=len(x)
        draw(i,root)
        i+=1
        if lx>=1:
            i = preordertraversal(G,x[0],i)
        if lx==2:
            i = preordertraversal(G,x[1],i)
        return i

    ino=preordertraversal(G,root)

    file=pics.objects.filter(name="preorder")
    files = []
    for a in file:
        files.append("." + a.frame.url)
    figure2 = io.BytesIO()
    images = [Image.open(file) for file in files]
    images[0].save(figure2, save_all=True, append_images=images[1:], duration=2000, loop=0,format="PNG")
    content_file = ImageFile(figure2)
    plot_instance1 = pics(name="preorder",fno=ino+2)
    plot_instance1.gif.save(output_gif, content_file)
    plot_instance1.save()
