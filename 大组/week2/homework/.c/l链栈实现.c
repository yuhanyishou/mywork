#include<stdio.h>
#include<stdbool.h>
#include<malloc.h>
#define ERROR 0
#define OK 1
typedef int elementype;
//������ջ�ڵ�ṹ��
typedef struct stacknode {
	elementype data;
	struct stacknode* next;
}stacknode;
//�������ṹ��
typedef struct stack {
	stacknode* top;  //ջ��ָ��
	int count;    //������

}Linkstack;
//��ʼ����ջ
int InitStack(Linkstack* S) {
	S->top = NULL;
	S->count = 0;
	return OK;
}
//�ж�ջ�Ƿ�Ϊ��
int stackempty(Linkstack S) {
	if (S.top == NULL) {
		return 0;
	}
	else {
		return 1;
	}
}
//��ջ����
int Push(Linkstack* S, elementype e) {
	stacknode* p = (stacknode*)malloc(sizeof(stacknode));
	if (!p) return ERROR;
	p->data = e;
	p->next = S->top;
	S->top = p;
	S->count++;
	return OK;
}
//��ȡջ�ĳ���
int StackLength(Linkstack S) {
	return S.count;
}
//��ջ����
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
//����ջԪ��
void Traversestack(Linkstack S) {
	stacknode* p = S.top;
	printf("ջ�е�Ԫ�طֱ�Ϊ��");
	while (p) {
		printf("%d", p->data);
		p = p->next;
	}
	printf("\n");
}
//���ջ
void ClearStack(Linkstack* S) {
	stacknode* p = S->top;
	while (p) {
		stacknode* q = p;
		p = p->next;
		free(q);
	}
	S->top = NULL;
	S->count = 0;
	printf("�ɹ����");
}
//����ջ��Ԫ��
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
	printf("������һЩ��ջ�Ļ����������밴��Ӧ������ʵ����صĲ�����\n");
	printf("1.��ʼ����ջ\n2.��ջ\n3.����ջ��Ԫ��\n4.����ջԪ��\n5.��ȡջ�ĳ���\n6.��ջ\n7.���ջ\n8.�ж�ջ�Ƿ�Ϊ��\n9.�˳�\n");
	printf("����������Ҫ�Ĳ�����");
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
					printf("��ʼ���ɹ�\n");
				else
					printf("��ʼ��ʧ��\n");
				break;
			case 2:
				printf("����ջ��1,2,3,������Ԫ��Ϊ��\n");
				for (int i = 1; i <= 3; i++) {
					if (Push(&S, i) == OK) {
						printf("Ԫ�� %d ��ջ�ɹ�\t", i);
					}
					else {
						printf("��ջʧ��\n");
						
					}
				}
				break;
			case 3:
				if (GetTop(S, &e) == OK)
					printf("\nջ��Ԫ��Ϊ��%d\n", e);
				else
					printf("��ȡջ��ʧ�ܣ�ջ�գ�\n");
				break;
			case 4:
				Traversestack(S);
				break;
			case 5:
				printf("��ǰջ���ȣ�%d\n", StackLength(S));
				break;
			case 6:
				while (StackLength(S) > 0) {
					if (Pop(&S, &e) == OK) {
						printf("��ջԪ�أ�%d\t", e);
						printf("ʣ��ջ���ȣ�%d\n", StackLength(S));
					}
					else {
						printf("��ջʧ��\n");
					}
				}
				break;
			case 7:
				ClearStack(&S);
				break;

			case 8:
				if (stackempty(S) == 0) {
					printf("ջΪ��\n");
				}
				else {
					printf("ջδ�����");
				}
			case 9:
				break;
			}	
		}
	}
	
	return 0;
}
