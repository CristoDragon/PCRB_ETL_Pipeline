# Author: Dragon Xu

# Introduction
1. Phase 1:
    - Intro: This phase will involve developing and deploying machine learning models such as NLP parsers as well as implementing data scrubbing and matching techniques such as fuzzy matching,  which will help us to accurately match the address between PCRB’ data and the D&B data and correct various data limitations and gaps to improve confidence levels of the matches.
    - Goal: Achieve at least 90% accuracy in data matching across the internal PCRB dataset and external D&B dataset. 
    - Measure of success: The accuracy will be measured as the percentage of successfully matched data rows from all the data in both datasets.
    - Deliverables
        - A combined dataset stored in SQL Server sandbox schema
        - A complete pipeline to clean, process, and match two datasets
2. Phase 2: 
    - Intro: Once businesses are matched and locations are identified, we will ensure that the total payroll amounts are correctly matched to create a more detailed and accurate view of payroll distributions by location. This phase will also validate data to ensure it aligns with the organization’s goals of providing detailed insights into geographically-based insurance premiums.
    - Goal: Correctly distribute payroll amount to each location.
    - Measure of success: Still need discussion.
    - Deliverables
        - A combined dataset with payroll information correctly distributed to each location


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

