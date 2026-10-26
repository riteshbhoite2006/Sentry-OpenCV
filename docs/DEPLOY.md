# Sentry — AWS EC2 Deployment Guide

> **Audience:** Demo video presenter and competition judges.  
> **Goal:** One EC2 instance running both containers. Judges open `http://<EC2-IP>:8501`.

---

## Table of Contents

1. [Build and Run Locally](#1-build-and-run-locally)
2. [Create the S3 Bucket (optional)](#2-create-the-s3-bucket-optional)
3. [Create the IAM Role for EC2](#3-create-the-iam-role-for-ec2)
4. [Launch the EC2 Instance](#4-launch-the-ec2-instance)
5. [Configure the Security Group](#5-configure-the-security-group)
6. [Install Docker on EC2](#6-install-docker-on-ec2)
7. [Clone and Start the Application](#7-clone-and-start-the-application)
8. [Verify the Stack is Running](#8-verify-the-stack-is-running)
9. [Set a Billing Alarm](#9-set-a-billing-alarm)
10. [View Logs](#10-view-logs)
11. [Stop / Start / Destroy the Instance](#11-stop--start--destroy-the-instance)
12. [Environment Variable Reference](#12-environment-variable-reference)

---

## 1. Build and Run Locally

**What these files do (for the demo video):**

| File | Plain-language description |
|------|---------------------------|
| `Dockerfile.api` | Builds the FastAPI inspection engine. Uses `python:3.13-slim` (same version as local). Verifies OpenCV 5.x at build time — the build *fails hard* if the wrong version is installed. Creates a non-root user `sentry` so the container never runs as root. |
| `Dockerfile.streamlit` | Builds the Streamlit dashboard with the same base image and the same OpenCV check. |
| `docker-compose.yml` | Wires the two containers together on a private Docker network. Only port 8501 (Streamlit) is published to the host. The API (port 8000) is invisible from the internet. Both services restart automatically if the EC2 instance reboots. |
| `.dockerignore` | Prevents sensitive files (`.env`, `*.pem`, `.aws/`, `data/`) from ever entering a Docker image layer. |

```bash
# From the repository root

# 1. Build both images (first build ~3 min; subsequent builds use cache)
docker compose build

# 2. Start in the background
docker compose up -d

# 3. Check both containers are healthy
docker compose ps

# 4. Open the dashboard
#    http://localhost:8501

# 5. Stop the stack (containers and network are removed; images stay)
docker compose down
```

**Rebuild after a code change:**

```bash
docker compose build && docker compose up -d
```

---

## 2. Create the S3 Bucket (optional)

> Skip if you do not want result persistence. The application works fully without S3.

Replace `<BUCKET_NAME>` and `<AWS_REGION>` throughout (e.g. `sentry-results-2026`, `us-east-1`).

```bash
# Create the bucket
# VERIFY AGAINST AWS DOCS — bucket names must be globally unique and follow naming rules
aws s3api create-bucket \
  --bucket <BUCKET_NAME> \
  --region <AWS_REGION> \
  --create-bucket-configuration LocationConstraint=<AWS_REGION>

# Block all public access (inspection results must not be public)
aws s3api put-public-access-block \
  --bucket <BUCKET_NAME> \
  --public-access-block-configuration \
    "BlockPublicAcls=true,IgnorePublicAcls=true,BlockPublicPolicy=true,RestrictPublicBuckets=true"
```

---

## 3. Create the IAM Role for EC2

This role gives the EC2 instance credentials automatically via the instance metadata service.  
**No access keys are ever stored on the server or in any file.**

### 3a. Trust policy — save as `ec2-trust-policy.json`

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Principal": { "Service": "ec2.amazonaws.com" },
      "Action": "sts:AssumeRole"
    }
  ]
}
```

### 3b. Permissions policy — save as `sentry-permissions.json`

Replace `<BUCKET_NAME>` and `<AWS_ACCOUNT_ID>` (12-digit number from the AWS Console top-right).

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "S3ResultsPersistence",
      "Effect": "Allow",
      "Action": [
        "s3:PutObject",
        "s3:GetObject"
      ],
      "Resource": "arn:aws:s3:::<BUCKET_NAME>/inspections/*"
    },
    {
      "Sid": "CloudWatchLogsWrite",
      "Effect": "Allow",
      "Action": [
        "logs:CreateLogGroup",
        "logs:CreateLogStream",
        "logs:PutLogEvents",
        "logs:DescribeLogStreams"
      ],
      "Resource": "arn:aws:logs:*:<AWS_ACCOUNT_ID>:log-group:/sentry/*"
    }
  ]
}
```

### 3c. Create and attach

```bash
# Create the IAM role
# VERIFY AGAINST AWS DOCS
aws iam create-role \
  --role-name SentryEC2Role \
  --assume-role-policy-document file://ec2-trust-policy.json

# Create the permissions policy
# VERIFY AGAINST AWS DOCS
aws iam create-policy \
  --policy-name SentryPermissionsPolicy \
  --policy-document file://sentry-permissions.json

# Attach the policy to the role (replace <AWS_ACCOUNT_ID>)
aws iam attach-role-policy \
  --role-name SentryEC2Role \
  --policy-arn arn:aws:iam::<AWS_ACCOUNT_ID>:policy/SentryPermissionsPolicy

# Create an instance profile (required to attach a role to an EC2 instance)
aws iam create-instance-profile \
  --instance-profile-name SentryInstanceProfile

aws iam add-role-to-instance-profile \
  --instance-profile-name SentryInstanceProfile \
  --role-name SentryEC2Role
```

---

## 4. Launch the EC2 Instance

> Recommended type: **t3.medium** (2 vCPU, 4 GB RAM). Enough for real-time inference.  
> Recommended OS: **Amazon Linux 2023** or **Ubuntu 22.04 LTS**.

```bash
# Step 1: Find the latest Amazon Linux 2023 AMI in your region
# VERIFY AGAINST AWS DOCS — AMI IDs change per region and release
aws ec2 describe-images \
  --owners amazon \
  --filters "Name=name,Values=al2023-ami-*-x86_64" \
             "Name=state,Values=available" \
  --query "sort_by(Images,&CreationDate)[-1].ImageId" \
  --output text

# Step 2: Launch the instance
# Replace: <AMI_ID>, <KEY_PAIR_NAME>, <SECURITY_GROUP_ID>
# VERIFY AGAINST AWS DOCS
aws ec2 run-instances \
  --image-id <AMI_ID> \
  --instance-type t3.medium \
  --key-name <KEY_PAIR_NAME> \
  --security-group-ids <SECURITY_GROUP_ID> \
  --iam-instance-profile Name=SentryInstanceProfile \
  --block-device-mappings "[{\"DeviceName\":\"/dev/xvda\",\"Ebs\":{\"VolumeSize\":20,\"VolumeType\":\"gp3\"}}]" \
  --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=sentry-demo}]" \
  --count 1

# Note the InstanceId from the output, e.g. i-0abc123def456789a
```

---

## 5. Configure the Security Group

**Only two ports are opened.** Port 8000 (API) stays closed to the internet — it is only reachable inside the Docker network.

| Port | Protocol | Source | Purpose |
|------|----------|--------|---------|
| 22   | TCP | Your IP only (`<YOUR_IP>/32`) | SSH for setup and maintenance |
| 8501 | TCP | `0.0.0.0/0` | Streamlit dashboard (public) |

```bash
# Find your current public IP
curl -s https://checkip.amazonaws.com

# Allow SSH from your IP only (replace <YOUR_IP> and <SG_ID>)
aws ec2 authorize-security-group-ingress \
  --group-id <SG_ID> \
  --protocol tcp --port 22 \
  --cidr <YOUR_IP>/32

# Allow Streamlit from anywhere
aws ec2 authorize-security-group-ingress \
  --group-id <SG_ID> \
  --protocol tcp --port 8501 \
  --cidr 0.0.0.0/0
```

> **Note:** If you created the security group via the AWS Console when launching the instance, you can edit inbound rules there instead of using the CLI.

---

## 6. Install Docker on EC2

```bash
# SSH into the instance
ssh -i <YOUR_KEY.pem> ec2-user@<EC2_PUBLIC_IP>
```

**Amazon Linux 2023:**

```bash
sudo dnf update -y
sudo dnf install -y docker git
sudo systemctl enable --now docker
sudo usermod -aG docker ec2-user

# Log out and back in so the group membership takes effect
exit
ssh -i <YOUR_KEY.pem> ec2-user@<EC2_PUBLIC_IP>

# Install the Docker Compose plugin
# VERIFY AGAINST AWS DOCS — latest version string may differ
COMPOSE_VER=$(curl -fsSL https://api.github.com/repos/docker/compose/releases/latest \
  | grep '"tag_name"' | sed -E 's/.*"v?([^"]+)".*/\1/')
sudo curl -SL "https://github.com/docker/compose/releases/download/v${COMPOSE_VER}/docker-compose-linux-x86_64" \
  -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose
docker compose version   # confirm
```

**Ubuntu 22.04 LTS:**

```bash
sudo apt-get update
sudo apt-get install -y ca-certificates curl gnupg git
sudo install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg \
  | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] \
  https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" \
  | sudo tee /etc/apt/sources.list.d/docker.list
sudo apt-get update
sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-compose-plugin
sudo usermod -aG docker ubuntu
exit
ssh -i <YOUR_KEY.pem> ubuntu@<EC2_PUBLIC_IP>
docker compose version   # confirm
```

---

## 7. Clone and Start the Application

```bash
# Clone the repo
git clone https://github.com/<YOUR_GITHUB_USERNAME>/sentry.git
cd sentry

# Optional: create a .env file for S3 and rate-limit settings
# The EC2 IAM role provides S3 credentials automatically via the instance metadata.
# NEVER put AWS_ACCESS_KEY_ID or AWS_SECRET_ACCESS_KEY here.
cat > .env << 'EOF'
S3_BUCKET=<BUCKET_NAME>
RATE_LIMIT_PER_MINUTE=20
EOF

# Build images and start in the background
# First build downloads ~500 MB of base layers (takes ~5-10 min on a t3.medium)
docker compose up -d --build

# Watch startup logs
docker compose logs -f
# Press Ctrl+C to stop following logs; containers keep running
```

The dashboard will be live at `http://<EC2_PUBLIC_IP>:8501` once both containers report **healthy**.

---

## 8. Verify the Stack is Running

```bash
# Show container status and health
docker compose ps

# Call the API health endpoint from within the instance
curl http://localhost:8000/health
# Expected: {"status":"healthy","opencv_version":"5.0.0",...}

# Open the Streamlit dashboard from your laptop
# http://<EC2_PUBLIC_IP>:8501
```

---

## 9. Set a Billing Alarm

Prevents surprise charges if you forget to stop the instance after the demo.

```bash
# Step 1: Create an SNS topic for email alerts
# VERIFY AGAINST AWS DOCS
aws sns create-topic --name SentryBillingAlerts --output text --query TopicArn
# Outputs: arn:aws:sns:<REGION>:<ACCOUNT_ID>:SentryBillingAlerts
# Save this as <SNS_TOPIC_ARN>

# Step 2: Subscribe your email
aws sns subscribe \
  --topic-arn <SNS_TOPIC_ARN> \
  --protocol email \
  --notification-endpoint <YOUR_EMAIL>
# Check your inbox and confirm the subscription

# Step 3: Create the alarm (billing metrics only exist in us-east-1)
# VERIFY AGAINST AWS DOCS
aws cloudwatch put-metric-alarm \
  --alarm-name "SentryDemoBillingOver10USD" \
  --alarm-description "Monthly estimated charges exceed $10" \
  --namespace AWS/Billing \
  --metric-name EstimatedCharges \
  --dimensions Name=Currency,Value=USD \
  --statistic Maximum \
  --period 86400 \
  --threshold 10 \
  --comparison-operator GreaterThanThreshold \
  --evaluation-periods 1 \
  --alarm-actions <SNS_TOPIC_ARN> \
  --region us-east-1
```

---

## 10. View Logs

The API emits **JSON-structured logs** to stdout. Each line is one event.

**Stream all logs in real time:**

```bash
docker compose logs -f
```

**Show only the API:**

```bash
docker compose logs -f api
```

**Filter by event type (requires `jq`):**

```bash
# Install jq if needed
sudo dnf install -y jq          # Amazon Linux
sudo apt-get install -y jq      # Ubuntu

# Show only completed inspections (verdict + timing)
docker compose logs api | jq 'select(.msg == "inspection_complete") | {ts, verdict, processing_time_ms, request_id}'

# Show only S3 warnings
docker compose logs api | jq 'select(.level == "WARNING")'
```

**Log fields reference:**

| Field | Description |
|-------|-------------|
| `ts` | ISO-8601 timestamp |
| `level` | `INFO` or `WARNING` |
| `msg` | `"request"` (every HTTP call) or `"inspection_complete"` |
| `request_id` | UUID unique to this request; also returned in the HTTP response header `X-Request-Id` |
| `verdict` | One of `defect_found`, `no_defect`, `recapture_requested`, `human_review` |
| `processing_time_ms` | End-to-end latency in milliseconds |
| `remote_ip` | Client IP address |

> **Privacy note:** Image bytes, credentials, tokens, and S3 keys are **never** written to logs.

---

## 11. Stop / Start / Destroy the Instance

**Stop Docker containers only (instance keeps running, EBS billed):**

```bash
docker compose down
```

**Stop the EC2 instance (compute billing pauses; EBS ~$0.10/GB/month continues):**

```bash
aws ec2 stop-instances --instance-ids <INSTANCE_ID>
# The public IP changes on restart unless you assigned an Elastic IP
```

**Start it again:**

```bash
aws ec2 start-instances --instance-ids <INSTANCE_ID>
# Wait ~30 s, then get the new public IP:
aws ec2 describe-instances \
  --instance-ids <INSTANCE_ID> \
  --query "Reservations[0].Instances[0].PublicIpAddress" \
  --output text
# SSH in and run: cd sentry && docker compose up -d
```

**After the demo — full cleanup to stop all charges:**

```bash
# 1. Delete S3 bucket contents and the bucket itself
aws s3 rm s3://<BUCKET_NAME> --recursive
aws s3api delete-bucket --bucket <BUCKET_NAME>

# 2. Terminate the EC2 instance (deletes root volume too)
aws ec2 terminate-instances --instance-ids <INSTANCE_ID>
```

---

## 12. Environment Variable Reference

Set these in a `.env` file in the repository root on the EC2 instance, or pass them directly to `docker compose up`.

| Variable | Default | Description |
|----------|---------|-------------|
| `S3_BUCKET` | *(unset)* | S3 bucket name for result persistence. Leave unset to disable S3 entirely. |
| `RATE_LIMIT_PER_MINUTE` | `20` | Maximum `/inspect` requests per minute per client IP. |
| `MAX_FILE_SIZE_MB` | `20` | Maximum upload size in megabytes. |
| `SENTRY_API_URL` | `http://api:8000` | URL that Streamlit uses to reach the API. The Docker-internal default is correct for compose; change only if running the API separately. |

---

*Generated for the OpenCV AI Competition 2026 — Sentry Concrete Crack Inspection System.*  
*Commands marked **VERIFY AGAINST AWS DOCS** should be cross-checked against the [official AWS CLI documentation](https://docs.aws.amazon.com/cli/latest/reference/) before use, as service interfaces and defaults evolve.*
