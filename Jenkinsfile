pipeline {
    agent any
stages {
        stage('Clone Repository') {
            steps {
                git branch: 'main', url:'https://github.com/Triss11/kpi-aws.git'
            }
        }
        stage('Docker') {
            steps {
                script {
                    withCredentials([string(credentialsId: 'DOCKER_PASS', variable: 'DOCKER_PASS')]){
                        sh "docker login -u sohini11 -p ${DOCKER_PASS}"
                        }
                    }
                }
            }
        stage('Pull Docker Image') {
            steps {
                script {
                    sh "docker pull sohini11/test-images:1.0.18"
                    }
                }
            }
        stage('Docker deploy') {
            steps {
                sh 'docker run -v /home/ec2-user/layoutlm.pt:/usr/src/main/kpi-aws/layoutlm.pt -itd -p 5000:5000 sohini11/test-images:1.0.18'
            }
        }
    }
}
