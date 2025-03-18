# Author: Dragon Xu

# Introduction
The Pennsylvania Compensation Rating Bureau (PCRB) is a non-profit corporation formed in accordance with the insurance laws, serving as the licensed rating organization for workers’ compensation insurance in Pennsylvania and Delaware. Established to support and promote the efficient operation of the workers’ compensation system, the bureau is responsible for the collection, analysis, and dissemination of data related to workers’ compensation insurance. As a matter of fact, PCRB is in the process of transitioning to a data broker role, utilizing its unique advantages of the ownership of workers’ compensation insurance data to provide more comprehensive and insightful data analytics to insurance carriers. This project aims to enhance PCRB's capabilities of giving unique insurance data insights that support insurance carriers in understanding their market share, claims, and other information depending on different business contexts, by improving data matching accuracy and alignment with insurance industry needs, supporting the organization's goal of becoming a leading provider of data-driven insights in the insurance sector. To achieve this, our team will implement a structured two-phase approach involving using machine learning models to accurately match the internal PCRB dataset and the external D&B (Dun & Bradstreet) dataset in phase 1, precisely distribute the payroll amount to each individual location in phase 2, and possible go into using data visualization techniques to create interactive dashboards based on the combined dataset above, which will provide better insights of geographically based insurance premiums to insurance carriers.

# Problems to Solve
Currently, PCRB faces several challenges in providing accurate and meaningful data insights to insurance carriers. First, let’s introduce more details about the two datasets that we mentioned above. The internal PCRB dataset contains data collected from insurance carriers in Pennsylvania and Delaware, comprising both claim data (details on indemnity, medical amounts, and claim counts) and policy data (details on business name, payroll, policy year, and addresses), whereas the external D&B dataset contains 159 columns, comprising business identification data (name, address, NAIC, FEIN, etc), financial data (number of employees, sales volume, square footage, etc), and other data such as CEO name, manufacturing indicator, industry category, etc. The primary issue is that PCRB is unable to use the dashboards to provide accurate insights of geographically distributed insurance premiums due to the absence of an accurate and comprehensive dataset that combines the internal and external datasets, which is the main problem that we need to solve in phase 1. Although PCRB’s existing data includes total payroll and total claim amounts for companies (policy holders), it often does not get amounts broken down by individual locations for companies with multiple stores or branches. So breaking the aggregated payroll amount and claims data into specific locations is the main issue we are going to solve in phase 2. Last but not least, there is a need to enhance customer engagement and sentiment by providing better, more accurate data analytics. These limitations hinder PCRB's transition to a data broker, impacting its ability to deliver value-added services to its stakeholders, including insurance carriers who require precise data for decision-making and market analysis.

# Objectives
1. Phase 1: 

Achieve at least 90% accuracy in data matching across different sources, specifically the internal PCRB dataset and external D&B dataset. The accuracy will be measured as the percentage of successfully matched data rows from all the data in both datasets. This phase will involve developing and deploying machine learning models such as NLP parsers as well as implementing data scrubbing and matching techniques such as fuzzy matching,  which will help us to accurately match the address between PCRB data and the D&B data and correct various data limitations and gaps to improve confidence levels of the matches.

2. Phase 2: 

Once businesses are matched and locations are identified, we will ensure that the total payroll amounts are correctly matched to create a more detailed and accurate view of payroll distributions by location. This phase will also validate data to ensure it aligns with the organization’s goals of providing detailed insights into geographically-based insurance premiums. However, we still need discussions on what measure to use to define the success of phase 2. Making sure we provide as many details of the data as possible and keeping high data integrity at the same time.

# Approches
1. To achieve the project objectives, our team will adopt the following approaches:
First, we will establish a team contract to outline guidelines for communication and conflict resolution, ensuring any challenges that arise during the project are addressed effectively. Additionally, we will clearly define all final deliverables and project milestones, ensuring alignment on expectations and outcomes across the team. A detailed timeline will also be created and managed using the SmartSheet tool available in Microsoft Teams, with tasks distributed based on each member's expertise, promoting efficient workload management and timely project completion.

2. Challenges that we anticipated:
First, scope changes could occur if the client adjusts project requirements or expectations, necessitating the team to adapt both its approach and timeline. Second, workload balance will be crucial, as team members will need to manage project responsibilities alongside academic commitments, potentially posing scheduling challenges and added stress. Lastly, our limited industry experience in the insurance sector may require additional learning and adaptation, which could extend the time needed to fully understand project details during the initial stages.

3. Team skills and strengths
Michael: Proficient in SSMS, VS Code, data integration, data cleaning,data analysis and dashboard development.
Dragon: Experienced in SQL development, RDBMS implementation, data cleaning pipelines, data visualization, and utilizing machine learning models.
Nirvik: Skilled in handling PII, understanding business processes in finance and insurance, as well as quantitative data analysis, data engineering, and data visualization.
Karl: Expertise in ETL processes, statistical modeling, and Natural Language Processing (NLP).

# Solutions
1. Team Contract and Communication: Establish a team contract to facilitate smooth communication and conflict resolution throughout the project.

2. Define Deliverables and Milestones: Clearly define the project's final deliverables and set specific milestones to ensure timely progress.

3. Task Distribution and Timeline: Distribute tasks among team members based on their skills and expertise and create a timeline to achieve the project objectives.

4. Data Integration and Model Development: Utilize SQL Server for data storage, Azure DevOps for project management, and VS Code for development. Leverage domain knowledge in the insurance sector to guide data integration and model development.

5. Machine Learning and NLP Implementation: Develop machine learning models and NLP parsers to improve data matching and address matching accuracy.

6. Quality Assurance and Testing: Conduct rigorous data quality checks and refine matching algorithms to minimize errors.

# Conclusion
This capstone project represents a critical step in PCRB's strategic shift to becoming a data broker. By enhancing data accuracy and providing meaningful insights to insurance carriers, PCRB will support its organizational goals, improve customer engagement, and drive revenue growth. A successful implementation of this project will enable PCRB to meet its strategic objectives and deliver significant value to its stakeholders.


# Team Best Practices
1. Project Management
    - Create corresponding work items in Azure DevOps(Boards/Boards) based on the action items that we confirmed during the internal team meeting
        - For most of development type of work, create the work item as "Enhancement". For debugging type of work, create the work item as "Bug"
        - Put a detailed description in the work items that you created, so that others (as well as yourself) can easily understand what you are working on. Include deliverables too if applied.
        - Set "Application" to "Actuarial Department - General" from the drop down menu for all work items
        - Set severity and due date fields based on your own situation, and other fields in the work items can remain as default
        - Link your commits to a specific work item that you created, so that others can see the coding part of your job. There are multiple ways to do this, and one easy way is to use "#XXXX" in your commit message, where XXXX is your work item ID. After you push your commits, you will be able to see the commit that you associated with the work item
    - There are multiple stages to categorize the current status of your work items, some common ones that we use are
        - "Being Verified": you are verifying all the information in the work item is accurate
        - "Issue Verified": you have verified all the information in the work item is accurate
        - "In Progress": you are in the process of completing this work item
        - "Testing": you have completed coding but are testing your code to makes sure it executes correctly
        - "Resolved": you completed developing and testing your code, and this work item is done


2. Coding Standard
    - Documentation
        - Documentation is very important for your teammates, your client, and even yourself in the future to understand your work. So be sure to document as much as possible while you are coding (insufficient documentation has been proved to be a common obstacles for software engineers)
        - Comment your methods or functions in a standard way. I suggest installing the extension called "Better Comments", which can automatically generate best-practice comment structure for your method.
        - Document all the assumptions that you make while coding, either they are small or big, temporary or permanent.
        - Other common documentation includes: dependencies, general workflow, requirements (what the users can and cannot not do)
    
    - Development
        - Always put your code to your own branch, and then merge to the main branch if needed. Do not push directly to the main branch.
        - Only push necessary files into the repo (.gitignore could come in handy)
        - Write a good commit message (follow the best practice)
        - Fully test your code before pushing to your own branch

