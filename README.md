# opensource_final_12.16: 최종 제출
***
## Project Title 
Voting classifier을 통한 Tumor classification


***
## Motivation
Voting classifier을 사용하면 각각의 classifier 정확도보다 높아진다는 것을 이용하여
Tumor classification의 정확도를 높였습니다.


***
## Build status

### Voting classifier을 구성하는 6개의 classifier는 다음과 같습니다.
* ExtraTreesClassifier
* RandomForestClassifier
* KNeighborsClassifier
* SVC
* HistGradientBoostingClassifier
* NuSVC


***
## Code style

### 각 classifier의 최적 하이퍼 파라미터를 다음과 같이 조정해 각각의 정확도를 높였습니다.

    etc=ExtraTreesClassifier(random_state=0,n_estimators=1000,n_jobs=-1,criterion='log_loss')
    rfc=RandomForestClassifier(random_state=0,n_jobs=-1,n_estimators=6000,bootstrap=False,criterion='entropy',max_features='log2')
    knn=KNeighborsClassifier(n_jobs=-1,n_neighbors=1,metric='manhattan')
    svc=SVC(gamma=0.01,C=1000,probability=True)
    hgbc = HistGradientBoostingClassifier(random_state=0,max_iter=800, learning_rate=0.1 )
    nsvc=NuSVC(random_state=0,probability=True, nu=0.05) 
    
### Voting Classifier의 코드입니다. 
### Soft, Hard Voting 중 Hard Voting을 사용했습니다.

    hard_vote  = VotingClassifier(models, voting='hard')
    hard_vote.fit(X_train, y_train)
    y_pred = hard_vote.predict(X_test)


***
## Screenshots

### 각 classifier의 개별 정확도와 voting classifier을 사용한 전체 정확도는 다음과 같습니다.
<img width="483" alt="개별정확도와 전체정확도 (2제출)" src="https://user-images.githubusercontent.com/115199282/207026033-59a18bf2-3664-43b8-b009-3c4f63f46297.png">





***
## Installation 안내: 다음의 패키지를 설치해야합니다.
* pip install scikit-learn
* pip install scikit-image

