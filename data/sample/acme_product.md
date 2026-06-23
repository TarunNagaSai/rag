# Acme Cloud — Product & SSO Notes

## Single Sign-On (SSO)
Acme Cloud supports SAML 2.0 and OpenID Connect. Enterprise (Platinum and Gold) plans
include SSO. SCIM user provisioning is available on Platinum only. The most common SSO
issues involve Azure AD (Entra ID) SAML assertion mapping and clock skew.

## Add-ons
- GraphQL Analytics (SKU AN-220): advanced query analytics, Platinum-eligible.
- Audit Log Export (SKU AL-110): streams audit events; required for SOC 2 customers.

## Plans
- Platinum: SSO + SCIM + all add-ons, 24/7 support, 99.95% SLA.
- Gold: SSO (no SCIM), business-hours support, 99.9% SLA.
- Silver: no SSO, email support, 99.5% SLA.

## Support Engineers
Marcus Lee specializes in identity and SSO. Dana Cho handles billing and provisioning.
Escalations route to the on-call lead.
