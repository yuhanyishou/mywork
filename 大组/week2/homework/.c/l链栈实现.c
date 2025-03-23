#include<stdio.h>
#include<stdbool.h>
#include<malloc.h>
#define ERROR 0
#define OK 1
typedef int elementype;
//定义链栈节点结构体
typedef struct stacknode {
	elementype data;
	struct stacknode* next;
}stacknode;
//定义链结构体
typedef struct stack {
	stacknode* top;  //栈顶指针
	int count;    //计数器

}Linkstack;
//初始化空栈
int InitStack(Linkstack* S) {
	S->top = NULL;
	S->count = 0;
	return OK;
}
//判断栈是否为空
int stackempty(Linkstack S) {
	if (S.top == NULL) {
		return 0;
	}
	else {
		return 1;
	}
}
//入栈操作
int Push(Linkstack* S, elementype e) {
	stacknode* p = (stacknode*)malloc(sizeof(stacknode));
	if (!p) return ERROR;
	p->data = e;
	p->next = S->top;
	S->top = p;
	S->count++;
	return OK;
}
//获取栈的长度
int StackLength(Linkstack S) {
	return S.count;
}
//出栈操作
int Pop(Linkstack* S, elementype* e) {
	if (S->top == NULL) {
		return ERROR;
	}
	else {
		stacknode* p = S->top;
		*e = p->data;
		S->top = p->next;
		free(p);
		S->count--;
		return OK;
	}
}
//遍历栈元素
void Traversestack(Linkstack S) {
	stacknode* p = S.top;
	printf("栈中的元素分别为：");
	while (p) {
		printf("%d", p->data);
		p = p->next;
	}
	printf("\n");
}
//清空栈
void ClearStack(Linkstack* S) {
	stacknode* p = S->top;
	while (p) {
		stacknode* q = p;
		p = p->next;
		free(q);
	}
	S->top = NULL;
	S->count = 0;
	printf("成功清空");
}
//返回栈顶元素
int GetTop(Linkstack S, elementype* e) {
	if (S.top == NULL) {
		return ERROR;
	}
	else {
		*e = S.top->data;
		return OK;
	}
}
int main() {
	printf("下面是一些链栈的基本操作，请按对应的数字实现相关的操作：\n");
	printf("1.初始化空栈\n2.入栈\n3.返回栈顶元素\n4.遍历栈元素\n5.获取栈的长度\n6.出栈\n7.清空栈\n8.判断栈是否为空\n9.退出\n");
	printf("请输入你想要的操作：");
	int count = 0;
	Linkstack S;
	elementype e;
	while (1) {
		int n = 0;
		int a = 0;
		scanf_s("%d", &n);
		if (n == 9) {
			break;
		}
		else {
			switch (n)
			{
			case 1:
				if (InitStack(&S) == OK)
					printf("初始化成功\n");
				else
					printf("初始化失败\n");
				break;
			case 2:
				printf("以入栈“1,2,3,”三个元素为例\n");
				for (int i = 1; i <= 3; i++) {
					if (Push(&S, i) == OK) {
						printf("元素 %d 入栈成功\t", i);
					}
					else {
						printf("入栈失败\n");
						
					}
				}
				break;
			case 3:
				if (GetTop(S, &e) == OK)
					printf("\n栈顶元素为：%d\n", e);
				else
					printf("获取栈顶失败（栈空）\n");
				break;
			case 4:
				Traversestack(S);
				break;
			case 5:
				printf("当前栈长度：%d\n", StackLength(S));
				break;
			case 6:
				while (StackLength(S) > 0) {
					if (Pop(&S, &e) == OK) {
						printf("出栈元素：%d\t", e);
						printf("剩余栈长度：%d\n", StackLength(S));
					}
					else {
						printf("出栈失败\n");
					}
				}
				break;
			case 7:
				ClearStack(&S);
				break;

			case 8:
				if (stackempty(S) == 0) {
					printf("栈为空\n");
				}
				else {
					printf("栈未被清空");
				}
			case 9:
				break;
			}	
		}
	}
	
	return 0;
}
