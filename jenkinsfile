pipeline{
    agent any

    stages{
        stage("Cloning Github repo to Jenkins"){
            steps{
                script{
                    echo "Cloning the repo to jenkins ....................."
                    checkout scmGit(branches: [[name: '*/main']], extensions: [], userRemoteConfigs: [[credentialsId: 'github-token', url: 'https://github.com/amar8737/Hotel_Reservation_Prediction.git']])
                }
            }
        }
    }
}